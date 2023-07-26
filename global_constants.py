from collections import defaultdict
import numpy as np

samp_freq = 25

region_map = defaultdict(lambda: np.nan)
region_map["ManRO"] = 0.0
region_map["ManRB"] = 1.0
region_map["ManRL"] = 2.0
region_map["ManAB"] = 3.0
region_map["ManAL"] = 4.0
region_map["ManLO"] = 5.0
region_map["ManLB"] = 6.0
region_map["ManLL"] = 7.0
region_map["MaxRO"] = 8.0
region_map["MaxRB"] = 9.0
region_map["MaxRL"] = 10.0
region_map["MaxAB"] = 11.0
region_map["MaxAL"] = 12.0
region_map["MaxLO"] = 13.0
region_map["MaxLB"] = 14.0
region_map["MaxLL"] = 15.0


region_names = ['ManRO', 'ManRB', 'ManRL', 'ManAB', 'ManAL', \
    'ManLO', 'ManLB', 'ManLL', 'MaxRO', 'MaxRB', 'MaxRL', 'MaxAB', \
    'MaxAL', 'MaxLO', 'MaxLB', 'MaxLL']


region_map_merge = defaultdict(lambda: np.nan)
region_map_merge["ManRO"] = 0.0
region_map_merge["ManRL"] = 0.0
region_map_merge["ManRB"] = 1.0
region_map_merge["MaxRB"] = 1.0
region_map_merge["ManAB"] = 2.0
region_map_merge["MaxAB"] = 2.0
region_map_merge["ManAL"] = 3.0
region_map_merge["ManLO"] = 4.0
region_map_merge["ManLL"] = 4.0
region_map_merge["ManLB"] = 5.0
region_map_merge["MaxLB"] = 5.0
region_map_merge["MaxRO"] = 6.0
region_map_merge["MaxRL"] = 6.0
region_map_merge["MaxAL"] = 7.0
region_map_merge["MaxLO"] = 8.0
region_map_merge["MaxLL"] = 8.0


merged_region_names = ['ManRO/ManRL', 'ManRB/MaxRB', 'ManAB/MaxAB', 'ManAL', 'ManLO/ManLL', \
    'ManLB/MaxLB', 'MaxRO/MaxRL', 'MaxAL', 'MaxLO/MaxLL']


# region_map_merge = defaultdict(lambda: np.nan)
# region_map_merge["ManRO"] = 0.0
# region_map_merge["ManRL"] = 0.0
# region_map_merge["ManRB"] = 1.0
# region_map_merge["ManAB"] = 2.0
# region_map_merge["ManAL"] = 3.0
# region_map_merge["ManLO"] = 4.0
# region_map_merge["ManLL"] = 4.0
# region_map_merge["ManLB"] = 5.0
# region_map_merge["MaxRO"] = 6.0
# region_map_merge["MaxRL"] = 6.0
# region_map_merge["MaxRB"] = 7.0
# region_map_merge["MaxAL"] = 8.0
# region_map_merge["MaxAB"] = 9.0
# region_map_merge["MaxLO"] = 10.0
# region_map_merge["MaxLL"] = 10.0
# region_map_merge["MaxLB"] = 11.0


# merged_region_names = ['ManRO/ManRL', 'ManRB', 'ManAB', 'ManAL', 'ManLO/ManLL', \
#     'ManLB', 'MaxRO/MaxRL', 'MaxRB', 'MaxAL', 'MaxAB', 'MaxLO/MaxLL', 'MaxLB']

left_handed_patients = ['P1', 'P5']


days_not_labeled = {
        'P1': [9, 16, 25, 29, 32, 33], 
        'P2': [1, 3, 11, 16, 25], 
        'P3': [4, 9, 11, 15, 16, 19, 24, 25, 26, 29, 34, 36], 
        'P4': [3, 5, 7, 10, 13, 16, 20, 38, 40], 
        'P5': [2, 8, 12, 19, 20, 21, 26, 27, 29, 30, 32, 35], 
        'P6': [2, 8,  10, 11, 12, 13, 16, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 34, 35, 39],
        'P7': [6, 7, 12, 14, 16, 17, 18, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40],
        'P8': [1, 2, 8, 9, 11, 12, 15, 25, 26, 28, 30, 31, 32, 34, 38, 39],
        'P9': [2, 3, 4, 6, 7, 10, 14, 18, 19, 21, 22],
        'P10': [1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 20,  24, 30, 32, 37, 38],
        'P11': [6, 8, 9, 10, 16, 18, 19, 20, 21, 22, 24, 27, 28, 29, 30, 33, 35, 36],
        'P12': [3, 7, 8, 10, 11, 12, 13, 14, 21]
    }
