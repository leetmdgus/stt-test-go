from django.db import models


class UserInfo(models.Model):
    name = models.CharField("이름", max_length=50)
    sex = models.CharField("성별", max_length=50)
    tendency = models.CharField("성향", max_length=50)
    latest_information = models.CharField("최근정보", max_length=50)

    def __str__(self):
        return self.name


class OpenAITxt(models.Model):
    name = models.CharField("이름", max_length=50)
    openai_txt = models.TextField("openai텍스트", max_length=20000)

    def __str__(self):
        return f"{self.name}의 텍스트"


class Checklist(models.Model):
    name = models.CharField("이름", max_length=50)
    q_category = models.CharField("사회 / 신체 / 정신", max_length=50)
    question = models.CharField("질문", max_length=50)
    answer = models.CharField("대답", max_length=50)

    def __str__(self):
        return f"{self.name} - {self.q_category}"
