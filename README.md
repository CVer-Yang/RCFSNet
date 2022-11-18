# RCFSNet
Road Extraction From Satellite Imagery by Road Context and Full-Stage Feature

Abstract：Road extraction from satellite imagery is vital in a broad range of applications. However, extracting complete roads
is challenging due to the road occlusions caused by surroundings. This paper proposed an improved encoder-decoder network via
extracting road context and integrating full-stage features from satellite imagery, dubbed as RCFSNet. A multi-scale context
extraction (MSCE) module is designed to enhance the inference capabilities by introducing adequate road context; Multiple full-stage feature fusion (FSFF) modules in the skip connection are devised to provide accurate road structure information, and we devise a coordinate dual attention mechanism (CDAM)
to strengthen the representation of road features. Extensive experiments demonstrate that our RCFSNet achieved new state-of-the-art performance on the public road datasets. The results indicate that the road labels extracted by our method have preferable connectivity.


# Code
Our code is based on DLinkNet .

### Training
DeepGlobe
`python main_RCFSNet_road.py`

Massachusetts
`python main_RCFSNet_Mas.py`

### Evaluation
DeepGlobe
`python test_RCFSNet_ROAD.py`
Massachusetts
`python test_RCFSNet_Mas.py`
