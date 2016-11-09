import sqlite3

list_annotation = list()
# Format for saving: path x y w h
ann_format = "{}/{} {} {} {} {}"

conn = sqlite3.connect('aflw.sqlite')

fidQuery = 'SELECT face_id FROM Faces'
faceIDs = conn.execute(fidQuery)

for idx in faceIDs:

    fidQuery = 'SELECT file_id FROM Faces WHERE face_id = {}'.format(idx[0])
    imgID = conn.execute(fidQuery)
    imgID = [id for id in imgID]

    imgDataQuery = "SELECT db_id,filepath,width,height FROM FaceImages WHERE file_id = '{}'".format(imgID[0][0])
    fileID = conn.execute(imgDataQuery)
    fileID = [id for id in fileID]
    db_id = fileID[0][0]
    filepath = fileID[0][1]

    faceRectQuery = 'SELECT x,y,w,h FROM FaceRect WHERE face_id = {}'.format(idx[0])
    faceRect = conn.execute(faceRectQuery)
    faceRect = [id for id in faceRect]

    if len(faceRect)==0:
        continue
    
    x,y,w,h =  faceRect[0]

    list_annotation.append(ann_format.format(db_id,filepath,x,y,w,h))


with open("AFLW_ann.txt",'w') as f:
    f.writelines("%s\n" % line for line in list_annotation) 