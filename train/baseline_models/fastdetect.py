import requests

class FastDetect:
    def __init__(self):
        self.API_CODE = "sk-4d475457208fc256e2eace39dec34f17"
        self.url = "http://region-9.autodl.pro:21504/api/detect"
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + self.API_CODE
        }
        self.features = []
        
    def detect(self, text):
        data = {
            "detector": "fast-detect(gpt-neo-2.7b)",
            "text": text
        }
        response = requests.post(self.url, headers=self.headers, json=data)
        
        response = response.json()
        if response['code'] == 0:
            return response['data']['prob']
        else:
            return self.detect(text)