'''
topic: 爬取英雄联盟英雄皮肤数据
data: https://lol.qq.com/data/info-heros.shtml
'''

import requests
import asyncio
import os
from aiohttp import ClientSession
import aiohttp
import json
from datetime import datetime

#异步编程爬取图片数据
async def skins_downloader(semaphore, hero_id, hero_name):
    async with semaphore:
        # 皮肤对应接口
        url = 'https://game.gtimg.cn/images/lol/act/img/js/hero/{}.js'.format(hero_id)
        dir_name = 'E:/hanll/研究生/course/数据挖掘/skins/{}'.format(hero_name)
        # 检查 dir_name 是否已经存在
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        async with ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            async with session.get(url) as response:
                response = await response.read() # 请求到对应英雄的 url 地址
                # 遍历 JSON 数据中的 skins 列表
                for skin in json.loads(response)['skins']:
                    if skin['mainImg']:
                        img_url = skin['mainImg'] # 皮肤对应的url
                        # kda女团皮肤名带斜杠，replace掉
                        # 保存路径
                        path = os.path.join(dir_name, '{}.jpg'.format(skin['name'].replace('/', ''), ))
                        async with session.get(img_url) as skin_response:
                            with open(path, 'wb') as f:
                                print('\rDownloading [{:^10}] {:<20}'.format(hero_name, skin['name']), end='')
                                f.write(await skin_response.read())

#API接口 返回hero列表
def hero_list():
    return requests.get('https://game.gtimg.cn/images/lol/act/img/js/heroList/hero_list.js').json()['hero']

async def run():
    semaphore = asyncio.Semaphore(30) # 创建一次最多允许30个并发请求的信号量对象
    heroes = hero_list() # 英雄列表数据
    tasks = []
    # 遍历遍历英雄列表，为每个英雄创建一个下载英雄皮肤的异步任务
    # 并储存在 tasks 列表
    for hero in heroes:
        tasks.append(asyncio.ensure_future(skins_downloader(semaphore, hero['heroId'], hero['title'])))
    # 等待所有 tasks 列表中的异步任务完成
    await asyncio.wait(tasks)

if __name__ == '__main__':
    start_time = datetime.now()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
    loop.close()
    end_time = datetime.now()
    time_diff = (end_time - start_time).seconds
    print('\nTime cost: {}s'.format(time_diff))