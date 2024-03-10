from django.shortcuts import render
from openai import OpenAI
from google.cloud import vision
from google.oauth2 import service_account
from dotenv import load_dotenv
import json

load_dotenv()

def test(request):

    # https://self-development.info/python%E3%81%A7google-cloud-vision-api%E3%82%92%E5%88%A9%E7%94%A8%E3%81%99%E3%82%8B%E6%96%B9%E6%B3%95/
    credentials = service_account.Credentials.from_service_account_file('.gc.json')
    client = vision.ImageAnnotatorClient(credentials=credentials)

    # https://cloud.google.com/vision/docs/ocr?hl=ja#vision_text_detection_gcs-python
    # path = './b.png'
    # with open(path, "rb") as image_file:
    #     content = image_file.read()

    output = ""

    if request.method == 'POST' and request.FILES['docimage']:
        content = request.FILES['docimage'].file.read()

        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations

        fullText = texts[0].description
        print(fullText)

        # print("Texts:")
        # for text in texts:
        #     print(f'\n"{text.description}"')
        #     vertices = [
        #         f"({vertex.x},{vertex.y})" for vertex in text.bounding_poly.vertices
        #     ]
        #     print("bounds: {}".format(",".join(vertices)))


        client = OpenAI()
        schema = {
            "見積もり情報リスト": [
                {
                    "名称": "string",
                    "摘要": "string | null",
                    "数量": "string | null",
                    "単価": "string | null",
                    "行の合計金額": "string",
                }
            ]
        }
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": f"次の工事見積書からOCRでスキャンして取得した文字列から、名称、摘要、数量、単価、行の合計金額を抜き出して、JSON形式で出力してください。JSONのスキーマは次の通りです：{schema}"},
                {"role": "user", "content": fullText}
            ]
        )

        resultText = response.choices[0].message.content
        resultData = json.loads(resultText)
        # print(resultData)

        output += "[名称],[摘要],[数量],[単価],[行の合計金額]"
        for r in resultData["見積もり情報リスト"]:
            print(r)
            output += f"{r['名称']},{r['摘要']},{r['数量']},{r['単価']},{r['行の合計金額']}\n"

    data = {
        'output': output,
    }
    return render(request, 'test.html', data)
