# ==============================================================
#   Copyright (C) 2021 whubaichuan. All rights reserved.
#   function： Demo of voiceprint recognition among three people by a vanilla CNN method
#              This file is the processing of the data.
# ==============================================================
#   Create by whubaichuan at 2021.05.01
#   Version 1.0
#   whubaichuan [huangbaichuan@whu.edu.cn]
# ==============================================================

import os
 
image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
 
 
def list_images(basePath, contains=None):
    return list_files(basePath, validExts=image_types, contains=contains)
 
 
def list_files(basePath, validExts=None, contains=None):

    for (rootDir, dirNames, filenames) in os.walk(basePath):

        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            ext = filename[filename.rfind("."):].lower()

            if validExts is None or ext.endswith(validExts):

                imagePath = os.path.join(rootDir, filename)
                yield imagePath