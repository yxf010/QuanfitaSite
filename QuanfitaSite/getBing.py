import requests
import json

def getUrl():
	url = 'https://cn.bing.com/HPImageArchive.aspx?format=js&idx=0&n=1&mkt=zh-CN'
	response = requests.get(url)
	img = json.loads(response.text)
	img = img['images'][0]['url']
	print(img)
	return img


def getImageUrl():
	url_base = 'http://www.bing.com'
	url = getUrl()
	res_url = url_base + url
	print(res_url)
	return res_url

if __name__ == '__main__':
	getImageUrl()