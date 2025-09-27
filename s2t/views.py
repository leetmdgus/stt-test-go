from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.response import Response
from django.shortcuts import render, HttpResponse, redirect
import sys, subprocess
import os
import io
import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
from django.views.decorators.csrf import csrf_exempt
from tqdm.auto import tqdm

import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import noisereduce as nr

from pathlib import Path
# faster-whisper
from faster_whisper import WhisperModel
from rest_framework.decorators import api_view, permission_classes
from pathlib import Path
from django.http import FileResponse, Http404
from . import parse
from . import parseopen
from .models import UserInfo, OpenAITxt, Checklist, M_UserInfo
# Create your views here.
@api_view(['GET'])
def index(request):
    name = request.data.get('namdd')
    print(name)
    print(3)
    return HttpResponse("Communication start")
    


def check1():
    print("PY:", sys.executable)
    try:
        import webrtcvad
        print("webrtcvad OK:", webrtcvad.__file__)
    except Exception as e:
        print("webrtcvad import FAIL:", repr(e))
    print("ffmpeg OK" if subprocess.call(["ffmpeg","-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)==0 else "ffmpeg NOT FOUND")

try:
    import webrtcvad
    HAS_WEBRTCVAD = True
except Exception:
    HAS_WEBRTCVAD = False

# pyannote (optional)
DIAR_ENABLED = False
try:
    from pyannote.audio import Pipeline as DiarizationPipeline
    DIAR_ENABLED = True
except Exception:
    DIAR_ENABLED = False
        
        
def load_audio_ffmpeg(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load (m4a/mp3/wav/...) via pydub+ffmpeg and convert to mono 16k wav array."""
    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1).set_frame_rate(target_sr).set_sample_width(2)  # 16-bit PCM
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    return samples, target_sr

def denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    """Spectral gating noise reduction (uses first 0.5s as noise profile)."""
    n_head = int(0.5 * sr)
    noise_clip = audio[:max(n_head, 1)]
    reduced = nr.reduce_noise(y=audio, y_noise=noise_clip, sr=sr, prop_decrease=0.9, stationary=False)
    return np.clip(reduced, -1.0, 1.0)

@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None

def write_txt(segments: List[Segment], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            spk = f"[{seg.speaker}] " if seg.speaker else ""
            f.write(f"{spk}{seg.text}\n")

def write_srt(segments: List[Segment], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    def fmt_time(t):
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{fmt_time(seg.start)} --> {fmt_time(seg.end)}\n")
            spk = f"[{seg.speaker}] " if seg.speaker else ""
            f.write(f"{spk}{seg.text}\n\n")

def write_json(segments: List[Segment], path: str, meta: dict = None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"segments": [asdict(s) for s in segments], "meta": meta or {}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        

# ------------ WebRTC VAD (optional) ------------
def frame_generator(frame_dur_ms: int, audio: bytes, sample_rate: int):
    n = int(sample_rate * (frame_dur_ms / 1000.0) * 2)  # 16-bit -> 2 bytes
    for i in range(0, len(audio), n):
        yield audio[i:i+n]

def vad_collect_webrtc(sample_rate: int,
                       audio_float: np.ndarray,
                       aggressiveness: int = 2,
                       frame_ms: int = 30,
                       padding_ms: int = 300,
                       min_speech_ms: int = 300,
                       max_segment_s: float = 30.0) -> List[Tuple[float, float]]:
    if not HAS_WEBRTCVAD:
        raise RuntimeError("webrtcvad is not available")
    vad = webrtcvad.Vad(aggressiveness)
    pcm16 = (audio_float * 32767.0).astype(np.int16).tobytes()

    expected_len = int(sample_rate * (frame_ms / 1000.0) * 2)  # 16-bit => *2
    frames = list(frame_generator(frame_ms, pcm16, sample_rate))
    n_frames = len(frames)
    samples_per_frame = int(sample_rate * (frame_ms / 1000.0))

    state = "nospeech"
    pad_frames = int(padding_ms / frame_ms)
    ring = [False]*pad_frames
    ring_i = 0
    cur_start = None
    segs = []

    for i, frame in enumerate(frames):
        # 🔧 마지막 불완전 프레임은 패스 (webrtcvad는 10/20/30ms 정확 길이만 허용)
        if len(frame) != expected_len:
            break  # 또는 `continue`로 무시해도 됨

        try:
            is_speech = vad.is_speech(frame, sample_rate)
        except Exception:
            # 방어적으로 한 번 더 스킵
            continue

        ring[ring_i] = is_speech
        ring_i = (ring_i + 1) % pad_frames
        voiced = sum(ring) > (pad_frames // 2)

        if state == "nospeech" and voiced:
            state = "speech"
            cur_start = max(0, i - pad_frames)
        elif state == "speech" and not voiced:
            state = "nospeech"
            end_i = i
            dur_ms = (end_i - cur_start) * frame_ms
            if dur_ms >= min_speech_ms:
                start_s = (cur_start * samples_per_frame) / sample_rate
                end_s = (end_i * samples_per_frame) / sample_rate
                if end_s - start_s > max_segment_s:
                    t = start_s
                    while t < end_s:
                        e = min(t + max_segment_s, end_s)
                        segs.append((t, e)); t = e
                else:
                    segs.append((start_s, end_s))

    # tail 처리
    if state == "speech" and cur_start is not None:
        end_i = n_frames - 1
        start_s = (cur_start * samples_per_frame) / sample_rate
        end_s = (end_i * samples_per_frame) / sample_rate
        if end_s - start_s > 0.1:
            if end_s - start_s > max_segment_s:
                t = start_s
                while t < end_s:
                    e = min(t + max_segment_s, end_s)
                    segs.append((t, e)); t = e
            else:
                segs.append((start_s, end_s))
    return segs


# ------------ Silero VAD (fallback) ------------
def vad_collect_silero(audio_float: np.ndarray,
                       sr: int,
                       min_speech_ms: int = 300,
                       max_segment_s: float = 30.0) -> List[Tuple[float, float]]:
    """Return list of (start_s, end_s) using Silero VAD."""
    import torch
    model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', onnx=False, trust_repo=True)
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    wav = torch.from_numpy(audio_float).float()
    ts_list = get_speech_timestamps(
        wav, model, sampling_rate=sr, return_seconds=True,
        min_speech_duration_ms=min_speech_ms
    )
    spans = []
    for ts in ts_list:
        start_s, end_s = float(ts['start']), float(ts['end'])
        t = start_s
        while t < end_s:
            e = min(t + max_segment_s, end_s)
            spans.append((t, e)); t = e
    return spans



def run_diarization(audio_path: str, auth_token: Optional[str] = None):
    """Returns list of (start, end, speaker_label)."""
    if not DIAR_ENABLED:
        return []
    if auth_token is None:
        auth_token = os.getenv("HUGGINGFACE_TOKEN", None)
    pipeline = DiarizationPipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=auth_token,
    )
    diar = pipeline(audio_path)
    turns = []
    for speech_turn in diar.itertracks(yield_label=True):
        (segment, _, spk) = speech_turn
        turns.append((float(segment.start), float(segment.end), str(spk)))
    return turns

def assign_speakers_to_segments(segments: List[Segment], turns: List[Tuple[float, float, str]]):
    if not turns:
        return segments
    for seg in segments:
        best = None; best_overlap = 0.0
        for (s, e, spk) in turns:
            overlap = max(0.0, min(seg.end, e) - max(seg.start, s))
            if overlap > best_overlap:
                best_overlap = overlap; best = spk
        seg.speaker = best
    return segments



def torch_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def transcribe_with_whisper(audio: np.ndarray,
                            sr: int,
                            speech_spans: List[Tuple[float, float]],
                            model_size: str = "large-v3",
                            device: str = "cpu", #안전모드 고정 #auto
                            compute_type: str = "float16",
                            language: Optional[str] = None,
                            initial_prompt: Optional[str] = None) -> List[Segment]:
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments: List[Segment] = []
    for (start_s, end_s) in tqdm(speech_spans, desc="STT (Whisper)", unit="seg"):
        s_idx = int(start_s * sr); e_idx = int(end_s * sr)
        chunk = audio[s_idx:e_idx]
        segs, _ = model.transcribe(
            chunk,
            language=language,
            beam_size=5, best_of=5, patience=0.2,
            temperature=[0.0, 0.2, 0.4],
            vad_filter=False,
            initial_prompt=initial_prompt,
            condition_on_previous_text=False,
            word_timestamps=False,
        )
        merged_text = " ".join(s.text.strip() for s in segs if s.text.strip())
        if merged_text:
            segments.append(Segment(start=start_s, end=end_s, text=merged_text))

    # Merge near-adjacent segments
    merged = []
    for seg in segments:
        if merged and (seg.start - merged[-1].end) < 0.6:
            merged[-1].end = seg.end
            merged[-1].text = (merged[-1].text + " " + seg.text).strip()
        else:
            merged.append(seg)
    return merged

def check3():
    import webrtcvad, shutil
    print("webrtcvad OK")
    print("ffmpeg:", shutil.which("ffmpeg"))
    print("ffprobe:", shutil.which("ffprobe"))

# @csrf_exempt
# def s2tExecute(audio_name):
#     print(audio_name, " 파일 실행 시작")
#     APP_DIR = Path(__file__).resolve().parent  # s2t/ 를 가리킴
#     AUDIO_PATH = APP_DIR / "sample" / "recording.m4a"  # s2t/sample/recording.m4a
#     # ========= User Configuration =========
#     #AUDIO_PATH   = "sample/recording.m4a"   # 입력 오디오 경로(.m4a 포함)
#     OUTDIR       = APP_DIR/"outputs"                 # 결과 저장 디렉토리
#     LANGUAGE     = "ko"                      # 'ko' or 'auto'
#     MODEL_SIZE   = "large-v3"                # faster-whisper model size
#     DIARIZE      = False                     # 화자 분리 사용 여부
#     SPEAKER_LABELS = "Senior,Worker"         # 화자 라벨(등장순서 매핑)
#     DOMAIN_TERMS = "복약, 낙상, 보행보조기, 일상생활수행능력, 통증, 활력징후, 식사, 수면, 배뇨, 상처"

#     # diarization을 사용할 경우, 아래처럼 환경변수(HF 토큰)를 설정해 주세요.
#     # import os
#     # os.environ['HUGGINGFACE_TOKEN'] = "hf_..."  # <- pyannote 사용 시 필요
#     # =====================================
#     print("initialization complete")
#     # ======== RUN PIPELINE ========
#     os.makedirs(OUTDIR, exist_ok=True)

#     # 1) Load & preprocess
#     audio, sr = load_audio_ffmpeg(AUDIO_PATH, target_sr=16000)
#     audio = denoise(audio, sr)

#     # 2) VAD
#     try:
#         if HAS_WEBRTCVAD:
#             spans = vad_collect_webrtc(sample_rate=sr, audio_float=audio,
#                                     aggressiveness=2, frame_ms=30, padding_ms=30,
#                                     min_speech_ms=30, max_segment_s=30.0)
#         else:
#             raise RuntimeError("webrtcvad not available")
#     except Exception as e:
#         print(f"[VAD] Falling back to Silero: {e}")
#         spans = vad_collect_silero(audio_float=audio, sr=sr,
#                                 min_speech_ms=300, max_segment_s=30.0)

#     if not spans:
#         raise RuntimeError("No speech detected.")

#     print("아직 괜찮음1")

#     # 3) STT
#     language = None if (LANGUAGE or '').lower() == 'auto' else LANGUAGE
#     initial_prompt = "다음은 노인 돌봄 상담 대화의 전사입니다. 전문 용어 예시: " + DOMAIN_TERMS

#     # compute_type = "float16" if torch_cuda_available() else "int8"
#     compute_type = "int8"   # 안전 모드 고정

#     stt_segments = transcribe_with_whisper(audio=audio, sr=sr, speech_spans=spans,
#                                         model_size=MODEL_SIZE, language=language,
#                                         initial_prompt=initial_prompt,
#                                         compute_type=compute_type)
#     print("아직 괜찮음2")

#     # 4) Diarization (optional)
#     diar_turns = []
#     spk_map = None
#     if DIARIZE and DIAR_ENABLED:
#         diar_turns = run_diarization(AUDIO_PATH)
#         stt_segments = assign_speakers_to_segments(stt_segments, diar_turns)
#         # Map raw speaker IDs to user-friendly labels by first-appearance order
#         raw_labels = [t[2] for t in diar_turns]
#         uniq = []
#         for x in raw_labels:
#             if x not in uniq:
#                 uniq.append(x)
#         desired = [x.strip() for x in (SPEAKER_LABELS or "").split(",")]
#         spk_map = {raw: (desired[i] if i < len(desired) else raw) for i, raw in enumerate(uniq)}
#         for seg in stt_segments:
#             if seg.speaker in (spk_map or {}):
#                 seg.speaker = spk_map[seg.speaker]
                
#     print("아직 괜찮음3")

#     # 5) Save outputs
#     base = Path(AUDIO_PATH).stem
#     out_txt  = str(Path(OUTDIR) / f"{base}_2nd.txt")
#     out_srt  = str(Path(OUTDIR) / f"{base}_2nd.srt")
#     # out_json = str(Path(OUTDIR) / f"{base}_2nd.json")

#     meta = {
#         "audio": AUDIO_PATH,
#         "sample_rate": sr,
#         "language_hint": language or "auto",
#         "model": MODEL_SIZE,
#         "diarization": bool(diar_turns),
#         "speaker_map": spk_map,
#     }

#     write_txt(stt_segments, out_txt)
#     write_srt(stt_segments, out_srt)
#     # write_json(stt_segments, out_json, meta=meta)

#     print("Saved:")
#     print(" -", out_txt)
#     print(" -", out_srt)
#     # print(" -", out_json)
#     return out_txt


# s2tExecute(3)


@csrf_exempt
def transmit_file(request):
    print("DDD")
    APP_DIR = Path(__file__).resolve().parent
    AUDIO_PATH = APP_DIR / "sample" / "recording33.m4a"
    if not AUDIO_PATH.exists():
        raise Http404("파일이 없습니다")
    return FileResponse(open(AUDIO_PATH, "rb"), as_attachment=True, filename="recording33.m4a")


@csrf_exempt
def download_file(request):
    if request.method == "POST" and request.FILES.get("file"):
        f = request.FILES["file"]
        save_dir = Path(__file__).resolve().parent / "sample"
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / f.name
        with open(save_path, "wb+") as dest:
            for chunk in f.chunks():
                dest.write(chunk)
        return JsonResponse({"message": f"업로드 성공: {f.name}"})
    return JsonResponse({"error": "파일이 없음"}, status=400)

import subprocess
from pathlib import Path

def to_wav_16k_mono_pcm(src_path: Path) -> Path:
    """
    m4a/mp3 등 → wav(16kHz, mono, 16bit PCM)로 변환.
    ffmpeg가 PATH에 있어야 합니다.
    """
    dst_path = src_path.with_suffix(".wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src_path),
        "-ac", "1",           # mono
        "-ar", "16000",       # 16kHz
        "-c:a", "pcm_s16le",  # 16bit PCM
        str(dst_path),
    ]
    # 오류 메시지 확인을 위해 stderr 캡처
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 변환 실패:\n{proc.stderr.decode('utf-8', 'ignore')}")
    return dst_path


count = 0
@csrf_exempt
@api_view(['POST'])
def counseling_start(request):
    from openai import OpenAI
    import os
    global count
    count += 1
    name = request.data.get('name')
    sex = request.data.get('sex')
    tendency = request.data.get('tendency')
    latest_information = request.data.get('meta_data')
    # obj = UserInfo(
    #         name=name,
    #         sex=sex,
    #         tendency=tendency,
    #         latest_information=latest_information
    #     )
    # obj.save()
    print("상담사 이름 : ", count)
    print("이름  : ", name)
    print("성별  : ", sex)
    print("성향  : ", tendency)
    print("최근 정보  : ", latest_information)
    
    # if request.method == "POST" and request.FILES.get("file"):
    #     f = request.FILES["file"]
    #     save_dir = Path(__file__).resolve().parent / "sample"
    #     save_dir.mkdir(exist_ok=True)
    #     print("파일 이름:",f.name)
    #     used_path = save_dir / f.name
    #     save_path = to_wav_16k_mono_pcm(used_path)
    #     with open(save_path, "wb+") as dest:
    #         for chunk in f.chunks():
    #             dest.write(chunk)
    obj = M_UserInfo(
            m_name = str(count),
            name=name,
            sex=sex,
            tendency=tendency,
            latest_information=latest_information
        )
    obj.save()
    f = request.FILES["file"]      
    content = ''
    if not f:
        # 파일이 없으면 빈 문자열 저장
        content = ""
    else: 
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        save_dir = Path(__file__).resolve().parent / "sample"
        used_path = save_dir / f'{str(count)}_{f.name}'
        with open(used_path, "wb+") as dest:
                for chunk in f.chunks():
                    dest.write(chunk)
        
        
        with open(used_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",  # 또는 whisper-1, gpt-4o-mini-transcribe
                file=f,
                language="ko"
            )
        
        content = print("Openai text", resp.text)
    print("저장경로 : ", used_path)
    obj1 = OpenAITxt(
            name=name,
            openai_txt=content,
        )
    obj1.save()
    raw = request.data.get("data")
    
    counseling_data = json.loads(raw)
    print("상담 데이터 : ", counseling_data)
    json_data = dict()
    json_data['social'] = []
    json_data['physical'] = []
    json_data['mental'] = []
    dummy_d = ['society', 'body', 'mental']
    print("상담 데이터", counseling_data)
    print("타입" , type(counseling_data))
    for i in dummy_d:
        if i == 'society':
            for j in counseling_data[i]:
                tempt = dict()
                tempt['question'] = j
                tempt['score'] = int(counseling_data[i][j][-2])
                tempt['answer'] = counseling_data[i][j][:-3]
                json_data['social'].append(tempt)
                obj2 = Checklist(
                        name=name,
                        q_category = '사회',
                        question = tempt['question'],
                        answer = tempt['answer'])
                obj2.save()
                
        if i == 'body':
            for j in counseling_data[i]:
                tempt = dict()
                tempt['question'] = j
                tempt['score'] = int(counseling_data[i][j][-2])
                tempt['answer'] = counseling_data[i][j][:-3]
                json_data['physical'].append(tempt)
                obj2 = Checklist(
                        name=name,
                        q_category = '신체',
                        question = tempt['question'],
                        answer = tempt['answer'])
                obj2.save()
                
        if i == 'mental':
            for j in counseling_data[i]:
                tempt = dict()
                tempt['question'] = j
                tempt['score'] = int(counseling_data[i][j][-2])
                tempt['answer'] = counseling_data[i][j][:-3]
                json_data['mental'].append(tempt)
                obj2 = Checklist(
                        name=name,
                        q_category = '정신',
                        question = tempt['question'],
                        answer = tempt['answer'])
                obj2.save()
                                
    print("society",json_data['social'])
    print("body",json_data['physical'])
    print("mental",json_data['mental'])
    obj2 = Checklist(
            name=name,
            q_category = '',
            question = '',
            answer = '')
    obj2.save()
    print('content : ', content)
    print('txt', json_data)
    if content == None:
        content = ""
    
    llm_text = llm_execute(json_data, content)

    print(llm_text)
    return JsonResponse({"message": llm_text})
    
    
    
        
        
        
@csrf_exempt
def llm_execute(context, text):
    rtr = parseopen.generate_report(context, text)
    print(rtr)
    return rtr
    
    

# file: s2t/views.py
import os
from pathlib import Path
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt

from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

def _azure_stt(audio_file_path: Path, language: str = "ko-KR") -> str:
    load_dotenv()  # .env 읽기
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    speech_region = os.getenv("AZURE_SPEECH_REGION")
    
    if not speech_key or not speech_region:
        raise RuntimeError("AZURE_SPEECH_KEY/AZURE_SPEECH_REGION 미설정")

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    print(1)
    speech_config.speech_recognition_language = language
    print(2)
    audio_config = speechsdk.audio.AudioConfig(filename=str(audio_file_path))
    print(3)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    print(4)
    result = recognizer.recognize_once()
    print(5)

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        raise RuntimeError("인식 실패: NoMatch")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation = speechsdk.CancellationDetails(result)
        raise RuntimeError(f"취소됨: {cancellation.reason}, {cancellation.error_details}")
    raise RuntimeError(f"인식 실패: {result.reason}")

@csrf_exempt  # 프론트가 다른 도메인이면 CSRF 예외 필요 (프로덕션에선 토큰/인증 설계 권장)
def s2tExecute(request):
    if request.method == "POST" and request.FILES.get("file"):
        f = request.FILES["file"]
        save_dir = Path(__file__).resolve().parent / "sample"
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / f.name
        with open(save_path, "wb+") as dest:
            for chunk in f.chunks():
                dest.write(chunk)
    print("캬캬캬", save_path)

    text = _azure_stt(save_path, language=request.POST.get("lang", "ko-KR"))
    print(text)
    return HttpResponse(200)
    return text
    
    
    
    
    