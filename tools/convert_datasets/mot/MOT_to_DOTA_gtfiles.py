import os

gtpath = 'G:/大创项目/test_process_data/gt/'
gtpath_new = 'G:/大创项目/test_process_data/new_gt/'

filenames = os.listdir(gtpath)
filenames.sort(key=lambda x:int(x[:-4]))
print(filenames)
offset = 0
Maxframe = 0
for file in filenames:
    offset = Maxframe
    Maxframe = 0
    f = open(gtpath+file, "r")
    lines = f.readlines()  # 读取全部内容 ，并以列表方式返回
    for line in lines:
        gt = line.strip().split(',')
        frame_id, ins_id = map(int, gt[:2])
        if frame_id > Maxframe:
            Maxframe = frame_id
        bbox = list(map(float, gt[2:6]))
        ori_bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], bbox[0], bbox[1]+bbox[3]]
        ori_line = str(ori_bbox[0])+' '+str(ori_bbox[1])+' '+str(ori_bbox[2])+' '+str(ori_bbox[3])+' '+str(ori_bbox[4])+' '+str(ori_bbox[5])+' '+str(ori_bbox[6])+' '+str(ori_bbox[7])+' '+gt[7]+' '+'0'
        # print(ori_line)
        f = open(gtpath_new+'P'+str(frame_id+offset-1).zfill(4)+'.txt', 'a')
        f.write(ori_line)
        f.write('\n')
        f.close()
        #print(ori_bbox)