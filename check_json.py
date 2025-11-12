#!/usr/bin/env python3
import json

# Check the vifactcheck JSON structure
with open('/Users/haila/Library/CloudStorage/GoogleDrive-ladohaingan@gmail.com/My Drive/MyFile/fake-new-detection/src/data/json/news_data_vifactcheck_train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    print('Total entries:', len(data))
    if data:
        print('Keys:', list(data[0].keys()))
        print('Images type:', type(data[0].get('images')))
        if data[0].get('images'):
            print('Number of images:', len(data[0]['images']))
            if data[0]['images']:
                print('First image keys:', list(data[0]['images'][0].keys()))
                print('Sample image data:')
                for key, value in data[0]['images'][0].items():
                    print(f'  {key}: {value}')
