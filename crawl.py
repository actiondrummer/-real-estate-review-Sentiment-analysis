import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd
import json
#전국 3단계 -> 4단계->5단계 하위레벨 만들기
def deep_level(geo):
    # geohash 32레벨 0~9 총 10개 b~z 중 i,l,o 제외
    geohash_level = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'm', 'n',
                     'p', 'q', 'r','s', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    #리스트 생성
    next_level=[]
    for i in geo:
        for j in geohash_level:
            next_level.append(i+j)

    return next_level

def getaptid(geo):
    for i in geo:
        id_url = "https://apis.zigbang.com/v2/aparts/items?domain=zigbang&geohash="+i
        html=requests.get(id_url)
        #html의 내용들은 btye형이므로 str형으로 변경
        data=html.content.decode("utf-8")
        #str -> json형태로 변경
        data=json.loads(data)
        # json에서 필요한 부분을 '필요한 부분 이름' 으로 가져오면 list형
        # #list형을 반복문을 이용하여 id 추출
        id_list=[]
        for j in data['vrItems']:
            id_list.append(j['areaDanjiId'])
        for k in data['recommendItems']:
            id_list.append(k['areaDanjiId'])
        for p in data["items"]:
            id_list.append(p['areaDanjiId'])
        aptid = pd.DataFrame(data=id_list,columns=['id'])
        aptid.to_csv("./aptid/"+i+".csv")
    return True

def getreview(geo):
    for i in geo:
        df=pd.read_csv('aptid/'+i+'.csv',encoding='utf-8')
        for j in range(len(df)):
            n=df['id'].iloc[j]
            url = "https://apis.zigbang.com/property/apartments/"+str(n)+"/reviews/v1"
            html = requests.get(url)
            data=html.content.decode("utf-8")
            data = json.loads(data)
            # a = [data['summary']]
            # data['data'] += a
            review = pd.DataFrame(data['data'])
            review.to_csv("./review/"+i+"_"+str(n)+".csv", encoding="utf-8")

#geohash 리스트 ,전국 3단계
geo=['wyd','wye','wy6','wy7']
#4단계
geo_4=deep_level(geo)
#5단계
geo_5=deep_level(geo_4)
if getaptid(geo_5):
    getreview(geo_5)
