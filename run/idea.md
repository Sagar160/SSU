Can be implemented
    1. Augmentation: rotation to +- 90, +-180
    2. residual model cnn ❌
    3. flow matching
    4. training on higher upsample data
    5. now try to introduce sign loss: ✅
        - checking sign loss, means if sdf is +ve or -ve with gt
    6. give additional information: 
        - if its niebhour have -ve or +ve value 
    7. increase subdivsion level in eval with training on higher level ❌
implemented Done:
    2. residual model cnn --> not working leading more errors
    5. now try to introduce sign loss --> improve results
    7. increase subdivsion level in eval --> not imporving but having low standard deviation.

learning
    1. lower batch give the lower validation loss ✅




64 original SDF eval:
cd1: 900

128 original SDF eval:
cd1: 480.090, cd2: 13.275, f1: 0.858, nc: 0.972, ecd2: 0.025, ef1: 0.704