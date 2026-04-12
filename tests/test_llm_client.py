import openai

YANDEX_CLOUD_FOLDER = "b1goqif30h3m8fgniagj"
YANDEX_CLOUD_API_KEY = "AQVNwP94jCOD1Vu-NXrH_KkgUA2w78i9ywUHKb6F"
YANDEX_CLOUD_MODEL = "yandexgpt-5-lite/latest"

client = openai.OpenAI(
    api_key=YANDEX_CLOUD_API_KEY,
    base_url="https://ai.api.cloud.yandex.net/v1",
    project=YANDEX_CLOUD_FOLDER
)

response = client.responses.create(
    model=f"gpt://{YANDEX_CLOUD_FOLDER}/{YANDEX_CLOUD_MODEL}",
    temperature=0.3,
    instructions="Ты помощник. Ответь кратко.",
    input="Привет, как дела?",
    max_output_tokens=100
)

print(response.output_text)