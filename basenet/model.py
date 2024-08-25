def Model_factory(backbone, num_classes):
    if backbone == 'hourglass52':
        from basenet.hourglass import StackedHourGlass as Model
        model = Model(num_classes, 1)
        
    elif backbone == 'hourglass104':
        from basenet.hourglass import StackedHourGlass as Model
        model = Model(num_classes, 2)
        
    elif backbone == 'gaussnet':
        from basenet.gaussnet import StackedHourGlass as Model
        model = Model(num_classes, 2)
        
    elif backbone == 'gaussnet_cascade_2layers':
        from basenet.gaussnet_cascade import StackedHourGlass as Model
        model = Model(num_classes, 1)
        
    elif backbone == 'gaussnet_cascade':
        from basenet.gaussnet_cascade import StackedHourGlass as Model
        model = Model(num_classes, 2)
    
    elif backbone == 'gaussnet_cascade_4layers':
        from basenet.gaussnet_cascade import StackedHourGlass as Model
        model = Model(num_classes, 3)
    
    elif backbone == 'uesnet101_dcn':
        from basenet.resnet_dcn import get_uesnet101 as Model
        model = Model(num_classes)
        
    elif backbone == 'uesnet18_dcn':
        from basenet.resnet_dcn import get_uesnet18 as Model
        model = Model(num_classes)
        
    elif backbone == 'DLA_dcn':
        from basenet.DLA_dcn import get_pose_net as Model
        model = Model(num_classes)
    
    else:
        raise "Model import Error !! "
        
    return model



