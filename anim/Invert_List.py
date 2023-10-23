bone_mapMotorica = [0, 43, 44, 45, 46, 47, 48, 49, 50, 1,None,2,3,4,5,6,7,8,12,13,14,15,16,17,21,22,23,18,19,21,9,10,11,24,25,26,27,31,32,33,34,35,36,40,41,42,37,38,39,28,29,30] 

new_map = [None] * 51

for idx, b in enumerate(bone_mapMotorica):
    if b is not None:
        new_map[b] = idx

print(new_map)
