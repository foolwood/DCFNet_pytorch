import json

OTB2015 = json.load(open('OTB2015.json', 'r'))
videos = OTB2015.keys()

OTB2013 = dict()
for v in videos:
   if v in ['carDark', 'car4', 'david', 'david2', 'sylvester', 'trellis', 'fish', 'mhyang', 'soccer', 'matrix',
            'ironman', 'deer', 'skating1', 'shaking', 'singer1', 'singer2', 'coke', 'bolt', 'boy', 'dudek',
            'crossing', 'couple', 'football1', 'jogging_1', 'jogging_2', 'doll', 'girl', 'walking2', 'walking',
            'fleetface', 'freeman1', 'freeman3', 'freeman4', 'david3', 'jumping', 'carScale', 'skiing', 'dog1',
            'suv', 'motorRolling', 'mountainBike', 'lemming', 'liquor', 'woman', 'faceocc1', 'faceocc2',
            'basketball', 'football', 'subway', 'tiger1', 'tiger2']:
        OTB2013[v] = OTB2015[v]


json.dump(OTB2013, open('OTB2013.json', 'w'), indent=2) 

