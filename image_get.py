#!/usr/bin/python
# -*- coding: utf-8 -*-

import requests
from urllib import request


def get_verify():
    list_hashkey = []
    for i in range(0, 10000):
        r = requests.get("https://nebula-stage.momenta.cn/site_api/v1/captcha/")
        text = eval(r.text)
        hashkey = text['hashkey']
        list_hashkey.append(hashkey)
    print(list_hashkey)
    return list_hashkey


def get_image(list_h):
    count = 1
    for i in list_h:
        count_str = str(count)
        IMAGE_URL = "https://nebula-stage.momenta.cn/site_api/v1/captcha/image/" + i
        request.urlretrieve(IMAGE_URL, 'image_to_tag/' + count_str + '.png')
        count = count + 1


if __name__ == '__main__':
    list_hashkey = get_verify()
    get_image(list_hashkey)
