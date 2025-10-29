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
    2. Directly upsample 32 to 128 improvement ✅
    3. chnaging the embedding imporvement ✅--- Need to check further maybe not even needed.

Suggestion:
1. change the normalizing technique in evaluation
2. Imporve the FM sampling technique use eular midpoint.
3. fvdb_marching cube is have some artifacts on lower resolution.

Nissim Suggestion:
1 compute mc mesh 
2 for each vertex of the mc mesh compute signed distance to gt mesh with igl for instance 
3 plot histogram of the sdf 4 display the mesh with a color assigned to each vertex based on its sdf





64 original SDF eval:
cd1: 900

128 original SDF eval:
cd1: 480.090, cd2: 13.275, f1: 0.858, nc: 0.972, ecd2: 0.025, ef1: 0.704