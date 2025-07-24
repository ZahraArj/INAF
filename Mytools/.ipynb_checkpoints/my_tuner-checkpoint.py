from keras_tuner import HyperParameters

from keras_tuner import HyperParameters

# ---------------------------------------------------------
# Build model for Geo branch with tunable hyperparameters
# ---------------------------------------------------------
def build_model_geo(hp, model_combined2, combined_loss, optimizer):
    # Activation functions
    act_geo1 = hp.Choice('act_geo1', values=['relu', 'tanh', 'sigmoid'])
    act_geo2 = hp.Choice('act_geo2', values=['relu', 'tanh', 'sigmoid'])
    act_geo3 = hp.Choice('act_geo3', values=['relu', 'tanh', 'sigmoid'])
    act_geo4 = hp.Choice('act_geo4', values=['relu', 'tanh', 'sigmoid'])

    # Dropout rates
    dropout_rate_1 = hp.Float('dropout_rate_1', min_value=0.0, max_value=0.5, step=0.1)
    dropout_rate_2 = hp.Float('dropout_rate_2', min_value=0.0, max_value=0.5, step=0.1)

    # Encoder/decoder layer units
    e1 = hp.Int('e1', min_value=16, max_value=128, step=16)
    e2 = hp.Int('e2', min_value=32, max_value=256, step=32)
    e3 = hp.Int('e3', min_value=64, max_value=512, step=64)
    d3 = hp.Int('d3', min_value=64, max_value=512, step=64)
    d2 = hp.Int('d2', min_value=32, max_value=256, step=32)

    # LSTM parameters
    act_lstm1 = hp.Choice('act_lstm1', values=['tanh', 'sigmoid', 'relu'])
    act_lstm2 = hp.Choice('act_lstm2', values=['tanh', 'sigmoid', 'relu'])
    do_lstm_1 = hp.Float('do_lstm_1', min_value=0.0, max_value=0.5, step=0.1)
    do_lstm_2 = hp.Float('do_lstm_2', min_value=0.0, max_value=0.5, step=0.1)
    do_lstm_3 = hp.Float('do_lstm_3', min_value=0.0, max_value=0.5, step=0.1)
    do_lstm_4 = hp.Float('do_lstm_4', min_value=0.0, max_value=0.5, step=0.1)

    # Final activation
    act16 = hp.Choice('act16', values=['tanh', 'sigmoid', 'relu'])

    model = model_combined2(
        act_geo1=act_geo1, act_geo2=act_geo2, act_geo3=act_geo3, act_geo4=act_geo4,
        dropout_rate_1=dropout_rate_1, dropout_rate_2=dropout_rate_2,
        e1=e1, e2=e2, e3=e3, d3=d3, d2=d2,
        act_lstm1=act_lstm1, act_lstm2=act_lstm2,
        do_lstm_1=do_lstm_1, do_lstm_2=do_lstm_2, do_lstm_3=do_lstm_3, do_lstm_4=do_lstm_4,
        act16=act16
    )

    model.compile(loss=combined_loss, optimizer=optimizer, metrics=['mae'])
    return model

# ---------------------------------------------------------
# Build model for Geo Attention branch 
# ---------------------------------------------------------
def build_model_geo_att(hp, model_combined2, combined_loss, optimizer):
    act_geo1 = hp.Choice('act_geo1', values=['relu', 'tanh', 'sigmoid'])
    act_geo2 = hp.Choice('act_geo2', values=['relu', 'tanh', 'sigmoid'])
    act_geo3 = hp.Choice('act_geo3', values=['relu', 'tanh', 'sigmoid'])
    act_geo4 = hp.Choice('act_geo4', values=['relu', 'tanh', 'sigmoid'])

    dropout_rate_1 = hp.Float('dropout_rate_1', min_value=0.0, max_value=0.5, step=0.1)
    dropout_rate_2 = hp.Float('dropout_rate_2', min_value=0.0, max_value=0.5, step=0.1)

    e1 = hp.Int('e1', min_value=16, max_value=128, step=16)
    e2 = hp.Int('e2', min_value=32, max_value=256, step=32)
    e3 = hp.Int('e3', min_value=64, max_value=512, step=64)
    d3 = hp.Int('d3', min_value=64, max_value=512, step=64)
    d2 = hp.Int('d2', min_value=32, max_value=256, step=32)

    act_lstm1 = hp.Choice('act_lstm1', values=['tanh', 'sigmoid', 'relu'])
    act_lstm2 = hp.Choice('act_lstm2', values=['tanh', 'sigmoid', 'relu'])
    do_lstm_1 = hp.Float('do_lstm_1', min_value=0.0, max_value=0.5, step=0.1)
    do_lstm_2 = hp.Float('do_lstm_2', min_value=0.0, max_value=0.5, step=0.1)
    do_lstm_3 = hp.Float('do_lstm_3', min_value=0.0, max_value=0.5, step=0.1)
    do_lstm_4 = hp.Float('do_lstm_4', min_value=0.0, max_value=0.5, step=0.1)

    act16 = hp.Choice('act16', values=['tanh', 'sigmoid', 'relu'])

    model = model_combined2(
        act_geo1=act_geo1, act_geo2=act_geo2, act_geo3=act_geo3, act_geo4=act_geo4,
        dropout_rate_1=dropout_rate_1, dropout_rate_2=dropout_rate_2,
        e1=e1, e2=e2, e3=e3, d3=d3, d2=d2,
        act_lstm1=act_lstm1, act_lstm2=act_lstm2,
        do_lstm_1=do_lstm_1, do_lstm_2=do_lstm_2, do_lstm_3=do_lstm_3, do_lstm_4=do_lstm_4,
        act16=act16
    )

    model.compile(loss=combined_loss, optimizer=optimizer, metrics=['mae'])
    return model
# ---------------------------------------------------------
# Build model for LiDAR branch with tunable hyperparameters
# ---------------------------------------------------------
def build_model_lidar(hp, model_combined2, combined_loss, optimizer):
    # ResNet parameters
    filters2 = hp.Int('filters2', min_value=64, max_value=256, step=64)
    filters3 = hp.Int('filters3', min_value=128, max_value=512, step=128)
    filters4 = hp.Int('filters4', min_value=256, max_value=1024, step=256)
    dense_last = hp.Int('dense_last', min_value=500, max_value=2000, step=500)
    act_last = hp.Choice('act_last', values=['relu', 'tanh', 'sigmoid'])

    # LiDAR specific layers
    act_li = hp.Choice('act_li', values=['relu', 'tanh', 'sigmoid'])
    dropout4 = hp.Float('dropout4', min_value=0.0, max_value=0.5, step=0.1)
    d1_li = hp.Int('d1_li', min_value=256, max_value=1024, step=256)
    d2_li = hp.Int('d2_li', min_value=128, max_value=512, step=128)
    d3_li = hp.Int('d3_li', min_value=64, max_value=256, step=64)
    d4_li = hp.Int('d4_li', min_value=32, max_value=128, step=32)

    # LSTM parameters
    act_lstm1 = hp.Choice('act_lstm1', values=['tanh', 'sigmoid', 'relu'])
    act_lstm2 = hp.Choice('act_lstm2', values=['tanh', 'sigmoid', 'relu'])
    do_lstm_1 = hp.Float('do_lstm_1', min_value=0.0, max_value=0.5, step=0.1)
    do_lstm_2 = hp.Float('do_lstm_2', min_value=0.0, max_value=0.5, step=0.1)
    do_lstm_3 = hp.Float('do_lstm_3', min_value=0.0, max_value=0.5, step=0.1)
    do_lstm_4 = hp.Float('do_lstm_4', min_value=0.0, max_value=0.5, step=0.1)

    # Final activation
    act16 = hp.Choice('act16', values=['tanh', 'sigmoid', 'relu'])
    d_lstm_li1 = hp.Int('d_lstm_li1', min_value=32, max_value=128, step=32)

    model = model_combined2(
        filters2=filters2, filters3=filters3, filters4=filters4,
        dense_last=dense_last, act_last=act_last,
        act_li=act_li, dropout4=dropout4,
        d1_li=d1_li, d2_li=d2_li, d3_li=d3_li, d4_li=d4_li,
        d_lstm_li1=d_lstm_li1,
        act_lstm1=act_lstm1, act_lstm2=act_lstm2,
        do_lstm_1=do_lstm_1, do_lstm_2=do_lstm_2, do_lstm_3=do_lstm_3, do_lstm_4=do_lstm_4,
        act16=act16
    )

    model.compile(loss=combined_loss, optimizer=optimizer, metrics=['mae'])
    return model
# ---------------------------------------------------------
# Build model for LiDar Attention branch 
# ---------------------------------------------------------

def build_model_lidar_att(hp, model_combined2, combined_loss, optimizer):
    # Hyperparameter choices for activations
    act_lstm = hp.Choice('act_lstm', values=['tanh', 'sigmoid', 'relu'])
    act_li = hp.Choice('act_li', values=['relu', 'tanh', 'sigmoid'])

    # Fully connected layer sizes
    d1_li = hp.Int('d1_li', min_value=128, max_value=512, step=64)
    d2_li = hp.Int('d2_li', min_value=64, max_value=256, step=64)
    d3_li = hp.Int('d3_li', min_value=32, max_value=128, step=32)
    d4_li = hp.Int('d4_li', min_value=32, max_value=64, step=32)

    # Dropout values for LSTM
    do_lstm_1 = hp.Float('do_lstm_1', 0.0, 0.5, step=0.1)
    do_lstm_2 = hp.Float('do_lstm_2', 0.0, 0.5, step=0.1)
    do_lstm_3 = hp.Float('do_lstm_3', 0.0, 0.5, step=0.1)
    do_lstm_4 = hp.Float('do_lstm_4', 0.0, 0.5, step=0.1)

    # Build model instance
    model = model_combined2(
        act_li=act_li,
        act_lstm=act_lstm,
        d1_li=d1_li,
        d2_li=d2_li,
        d3_li=d3_li,
        d4_li=d4_li,
        do_lstm_1=do_lstm_1,
        do_lstm_2=do_lstm_2,
        do_lstm_3=do_lstm_3,
        do_lstm_4=do_lstm_4
    )

    # Compile with loss and optimizer
    model.compile(loss=combined_loss, optimizer=optimizer, metrics=['mae'])
    return model


# ---------------------------------------------------------
# Build model for combined Geo + LiDAR branch
# ---------------------------------------------------------
def build_model_all(hp, model_combined2, combined_loss, optimizer):
    # Geo branch
    act_geo1 = hp.Choice('act_geo1', values=['relu', 'tanh', 'sigmoid'])
    act_geo2 = hp.Choice('act_geo2', values=['relu', 'tanh', 'sigmoid'])
    act_geo4 = hp.Choice('act_geo4', values=['relu', 'tanh', 'sigmoid'])
    act_geo_lstm1 = hp.Choice('act_geo_lstm1', values=['tanh', 'sigmoid', 'relu'])
    act_geo_lstm2 = hp.Choice('act_geo_lstm2', values=['tanh', 'sigmoid', 'relu'])
    dropout_rate_1_geo = hp.Float('dropout_rate_1_geo', min_value=0.0, max_value=0.3, step=0.1)
    e1_geo = hp.Int('e1_geo', min_value=16, max_value=128, step=16)
    e2_geo = hp.Int('e2_geo', min_value=32, max_value=256, step=32)
    d3_geo = hp.Int('d3_geo', min_value=64, max_value=512, step=64)
    do_lstm_geo_3 = hp.Float('do_lstm_geo_3', min_value=0.0, max_value=0.3, step=0.1)
    do_lstm_geo_4 = hp.Float('do_lstm_geo_4', min_value=0.0, max_value=0.3, step=0.1)

    # LiDAR branch
    filters2 = hp.Int('filters2', min_value=64, max_value=256, step=64)
    filters3 = hp.Int('filters3', min_value=128, max_value=512, step=128)
    filters4 = hp.Int('filters4', min_value=256, max_value=1024, step=256)
    dense_last = hp.Int('dense_last', min_value=500, max_value=2000, step=500)
    act_last = hp.Choice('act_last', values=['relu', 'tanh', 'sigmoid'])
    act1_li = hp.Choice('act1_li', values=['relu', 'tanh', 'sigmoid'])
    act2_li = hp.Choice('act2_li', values=['relu', 'tanh', 'sigmoid'])
    dropout_rate_li = hp.Float('dropout_rate_li', min_value=0.0, max_value=0.3, step=0.1)
    d1_li = hp.Int('d1_li', min_value=256, max_value=1024, step=256)
    d2_li = hp.Int('d2_li', min_value=128, max_value=512, step=128)
    act_li_lstm3 = hp.Choice('act_li_lstm3', values=['tanh', 'sigmoid', 'relu'])
    act_li_lstm4 = hp.Choice('act_li_lstm4', values=['tanh', 'sigmoid', 'relu'])
    do_lstm_li_3 = hp.Float('do_lstm_li_3', min_value=0.0, max_value=0.3, step=0.1)
    do_lstm_li_4 = hp.Float('do_lstm_li_4', min_value=0.0, max_value=0.3, step=0.1)

    # Fusion and output
    d1_fuse = hp.Int('d1_fuse', min_value=16, max_value=64, step=16)
    act_fuse = hp.Choice('act_fuse', values=['relu', 'tanh', 'sigmoid'])
    dropout_rate_fuse = hp.Float('dropout_rate_fuse', min_value=0.0, max_value=0.3, step=0.1)
    d3_end = hp.Int('d3_end', min_value=8, max_value=64, step=8)
    d2_end = hp.Int('d2_end', min_value=4, max_value=32, step=4)
    act_end_1 = hp.Choice('act_end', values=['tanh', 'sigmoid', 'relu'])
    act_end_2 = hp.Choice('act_end', values=['tanh', 'sigmoid', 'relu'])

    model = model_combined2(
        act_geo1=act_geo1, act_geo2=act_geo2, act_geo4=act_geo4,
        dropout_rate_1_geo=dropout_rate_1_geo,
        e1_geo=e1_geo, e2_geo=e2_geo, d3_geo=d3_geo,
        act_geo_lstm1=act_geo_lstm1, act_geo_lstm2=act_geo_lstm2,
        do_lstm_geo_3=do_lstm_geo_3, do_lstm_geo_4=do_lstm_geo_4,
        filters2=filters2, filters3=filters3, filters4=filters4,
        dense_last=dense_last, act_last=act_last,
        act1_li=act1_li, act2_li=act2_li,
        dropout_rate_li=dropout_rate_li,
        d1_li=d1_li, d2_li=d2_li,
        act_li_lstm3=act_li_lstm3, act_li_lstm4=act_li_lstm4,
        do_lstm_li_3=do_lstm_li_3, do_lstm_li_4=do_lstm_li_4,
        d1_fuse=d1_fuse, act_fuse=act_fuse,
        dropout_rate_fuse=dropout_rate_fuse,
        d3_end=d3_end, d2_end=d2_end,
        act_end_1=act_end_1, act_end_2=act_end_2
    )

    model.compile(loss=combined_loss, optimizer=optimizer, metrics=['mae'])
    return model

# ---------------------------------------------------------
# Build model for combined Geo + LiDAR branch with attention
# ---------------------------------------------------------
def build_model_all_att(hp, model_combined2, combined_loss, optimizer):
    # ----------------------------- Geo Branch -----------------------------
    act_geo1 = hp.Choice('act_geo1', values=['relu', 'tanh', 'sigmoid'])
    act_geo2 = hp.Choice('act_geo2', values=['relu', 'tanh', 'sigmoid'])
    act_geo4 = hp.Choice('act_geo4', values=['relu', 'tanh', 'sigmoid'])

    act_geo_lstm1 = hp.Choice('act_geo_lstm1', values=['tanh', 'sigmoid', 'relu'])
    act_geo_lstm2 = hp.Choice('act_geo_lstm2', values=['tanh', 'sigmoid', 'relu'])

    dropout_rate_1_geo = hp.Float('dropout_rate_1_geo', min_value=0.0, max_value=0.3, step=0.1)

    e1_geo = hp.Int('e1_geo', min_value=16, max_value=128, step=16)
    e2_geo = hp.Int('e2_geo', min_value=32, max_value=256, step=32)
    d3_geo = hp.Int('d3_geo', min_value=64, max_value=512, step=64)

    do_lstm_geo_3 = hp.Float('do_lstm_geo_3', min_value=0.0, max_value=0.3, step=0.1)
    do_lstm_geo_4 = hp.Float('do_lstm_geo_4', min_value=0.0, max_value=0.3, step=0.1)

    # ----------------------------- LiDAR Branch -----------------------------
    filters2 = hp.Int('filters2', min_value=64, max_value=256, step=64)
    filters3 = hp.Int('filters3', min_value=128, max_value=512, step=128)
    filters4 = hp.Int('filters4', min_value=256, max_value=1024, step=256)

    dense_last = hp.Int('dense_last', min_value=500, max_value=2000, step=500)
    act_last = hp.Choice('act_last', values=['relu', 'tanh', 'sigmoid'])

    act1_li = hp.Choice('act1_li', values=['relu', 'tanh', 'sigmoid'])
    act2_li = hp.Choice('act2_li', values=['relu', 'tanh', 'sigmoid'])

    dropout_rate_li = hp.Float('dropout_rate_li', min_value=0.0, max_value=0.3, step=0.1)

    d1_li = hp.Int('d1_li', min_value=256, max_value=1024, step=256)
    d2_li = hp.Int('d2_li', min_value=128, max_value=512, step=128)

    act_li_lstm3 = hp.Choice('act_li_lstm3', values=['tanh', 'sigmoid', 'relu'])
    act_li_lstm4 = hp.Choice('act_li_lstm4', values=['tanh', 'sigmoid', 'relu'])

    do_lstm_li_3 = hp.Float('do_lstm_li_3', min_value=0.0, max_value=0.3, step=0.1)
    do_lstm_li_4 = hp.Float('do_lstm_li_4', min_value=0.0, max_value=0.3, step=0.1)

    # ----------------------------- Fusion & Output -----------------------------
    d1_fuse = hp.Int('d1_fuse', min_value=16, max_value=64, step=16)
    act_fuse = hp.Choice('act_fuse', values=['relu', 'tanh', 'sigmoid'])
    dropout_rate_fuse = hp.Float('dropout_rate_fuse', min_value=0.0, max_value=0.3, step=0.1)

    d3_end = hp.Int('d3_end', min_value=8, max_value=64, step=8)
    d2_end = hp.Int('d2_end', min_value=4, max_value=32, step=4)
    act_end_1 = hp.Choice('act_end_1', values=['tanh', 'sigmoid', 'relu'])
    act_end_2 = hp.Choice('act_end_2', values=['tanh', 'sigmoid', 'relu'])

    # ----------------------------- Build and Compile -----------------------------
    model = model_combined2(
        act_geo1=act_geo1, act_geo2=act_geo2, act_geo4=act_geo4,
        dropout_rate_1_geo=dropout_rate_1_geo,
        e1_geo=e1_geo, e2_geo=e2_geo, d3_geo=d3_geo,
        act_geo_lstm1=act_geo_lstm1, act_geo_lstm2=act_geo_lstm2,
        do_lstm_geo_3=do_lstm_geo_3, do_lstm_geo_4=do_lstm_geo_4,
        filters2=filters2, filters3=filters3, filters4=filters4,
        dense_last=dense_last, act_last=act_last,
        act1_li=act1_li, act2_li=act2_li,
        dropout_rate_li=dropout_rate_li,
        d1_li=d1_li, d2_li=d2_li,
        act_li_lstm3=act_li_lstm3, act_li_lstm4=act_li_lstm4,
        do_lstm_li_3=do_lstm_li_3, do_lstm_li_4=do_lstm_li_4,
        d1_fuse=d1_fuse, act_fuse=act_fuse,
        dropout_rate_fuse=dropout_rate_fuse,
        d3_end=d3_end, d2_end=d2_end,
        act_end_1=act_end_1, act_end_2=act_end_2
    )

    model.compile(loss=combined_loss, optimizer=optimizer, metrics=['mae'])
    return model

# def build_model_all(hp, model_combined2, combined_loss, optimizer):
    
#     # ________________________________________________________________Geo branch
#     act_geo1 = hp.Choice('act_geo1', values=['relu', 'tanh', 'sigmoid'])
#     act_geo2 = hp.Choice('act_geo2', values=['relu', 'tanh', 'sigmoid'])
#     act_geo3 = hp.Choice('act_geo3', values=['relu', 'tanh', 'sigmoid'])
#     act_geo4 = hp.Choice('act_geo4', values=['relu', 'tanh', 'sigmoid'])
    
#     act_geo_lstm1 = hp.Choice('act_geo_lstm1', values=['tanh', 'sigmoid', 'relu'])
#     act_geo_lstm2 = hp.Choice('act_geo_lstm2', values=['tanh', 'sigmoid', 'relu'])
    
#     dropout_rate_1_geo = hp.Float('dropout_rate_1_geo', min_value=0.0, max_value=0.3, step=0.1)
#     dropout_rate_2_geo = hp.Float('dropout_rate_2_geo', min_value=0.0, max_value=0.3, step=0.1)
    
#     e1_geo = hp.Int('e1_geo', min_value=16, max_value=128, step=16)
#     e2_geo = hp.Int('e2_geo', min_value=32, max_value=256, step=32)
#     e3_geo = hp.Int('e3_geo', min_value=64, max_value=512, step=64)
#     d3_geo = hp.Int('d3_geo', min_value=64, max_value=512, step=64)
#     d2_geo = hp.Int('d2_geo', min_value=32, max_value=256, step=32)
    
#     do_lstm_geo_1 = hp.Float('do_lstm_geo_1', min_value=0.0, max_value=0.3, step=0.1)
#     do_lstm_geo_2 = hp.Float('do_lstm_geo_2', min_value=0.0, max_value=0.3, step=0.1)
#     do_lstm_geo_3 = hp.Float('do_lstm_geo_3', min_value=0.0, max_value=0.3, step=0.1)
#     do_lstm_geo_4 = hp.Float('do_lstm_geo_4', min_value=0.0, max_value=0.3, step=0.1)

#     # ________________________________________________________________LiDAR branch
#     # _____________________________________________________resnet
#     filters2 = hp.Int('filters2', min_value=64, max_value=256, step=64)
#     filters3 = hp.Int('filters3', min_value=128, max_value=512, step=128)
#     filters4 = hp.Int('filters4', min_value=256, max_value=1024, step=256)
    
#     dense_last = hp.Int('dense_last', min_value=500, max_value=2000, step=500)   
#     act_last = hp.Choice('act_last', values=['relu', 'tanh', 'sigmoid'])
#     # _____________________________________________________Lidar
#     act1_li = hp.Choice('act1_li', values=['relu', 'tanh', 'sigmoid'])
#     act2_li = hp.Choice('act2_li', values=['relu', 'tanh', 'sigmoid'])
#     act3_li = hp.Choice('act3_li', values=['relu', 'tanh', 'sigmoid'])
#     act4_li = hp.Choice('act4_li', values=['relu', 'tanh', 'sigmoid'])
    
#     dropout_rate_li = hp.Float('dropout_rate_li', min_value=0.0, max_value=0.3, step=0.1)
    
#     d1_li = hp.Int('d1_li', min_value=256, max_value=1024, step=256)
#     d2_li = hp.Int('d2_li', min_value=128, max_value=512, step=128)
#     d3_li = hp.Int('d3_li', min_value=64, max_value=256, step=64)
#     d4_li = hp.Int('d4_li', min_value=32, max_value=128, step=32)
    
#     d1_lstm_li = hp.Int('d1_lstm_li', min_value=8, max_value=32, step=8)
    
#     act_li_lstm1 = hp.Choice('act_li_lstm1', values=['tanh', 'sigmoid', 'relu'])
#     act_li_lstm2 = hp.Choice('act_li_lstm2', values=['tanh', 'sigmoid', 'relu'])
#     act_li_lstm3 = hp.Choice('act_li_lstm3', values=['tanh', 'sigmoid', 'relu'])
#     act_li_lstm4 = hp.Choice('act_li_lstm4', values=['tanh', 'sigmoid', 'relu'])
    
#     do_lstm_li_1 = hp.Float('do_lstm_li_1', min_value=0.0, max_value=0.3, step=0.1)
#     do_lstm_li_2 = hp.Float('do_lstm_li_2', min_value=0.0, max_value=0.3, step=0.1)
#     do_lstm_li_3 = hp.Float('do_lstm_li_3', min_value=0.0, max_value=0.3, step=0.1)
#     do_lstm_li_4 = hp.Float('do_lstm_li_4', min_value=0.0, max_value=0.3, step=0.1)

#     # ______________________________________________________Fusion
#     d1_fuse = hp.Int('d1_fuse', min_value=16, max_value=64, step=16)
#     act_fuse = hp.Choice('act_fuse', values=['relu', 'tanh', 'sigmoid'])
#     dropout_rate_fuse = hp.Float('dropout_rate_fuse', min_value=0.0, max_value=0.3, step=0.1)
#     d3_end = hp.Int('d3_end', min_value=8, max_value=64, step=8)
#     d2_end = hp.Int('d2_end', min_value=4, max_value=32, step=4)
#     act_end_1 = hp.Choice('act_end', values=['tanh', 'sigmoid', 'relu'])
#     act_end_2 = hp.Choice('act_end', values=['tanh', 'sigmoid', 'relu'])

#     model = model_combined2(
#         act_geo1=act_geo1, act_geo2=act_geo2, act_geo3=act_geo3, act_geo4=act_geo4,
#         dropout_rate_1_geo=dropout_rate_1_geo, dropout_rate_2_geo=dropout_rate_2_geo,
#         e1_geo=e1_geo, e2_geo=e2_geo, e3_geo=e3_geo, d3_geo=d3_geo, d2_geo=d2_geo,
#         act_geo_lstm1=act_geo_lstm1, act_geo_lstm2=act_geo_lstm2, 
#         do_lstm_geo_1=do_lstm_geo_1, do_lstm_geo_2=do_lstm_geo_2, do_lstm_geo_3=do_lstm_geo_3, do_lstm_geo_4=do_lstm_geo_4,
#         filters2=filters2, filters3=filters3, filters4=filters4,
#         dense_last=dense_last, act_last=act_last,
#         act1_li=act1_li, act2_li=act2_li, act3_li=act3_li, act4_li=act4_li,
#         dropout_rate_li=dropout_rate_li,
#         d1_li=d1_li, d2_li=d2_li, d3_li=d3_li, d4_li=d4_li,
#         d1_lstm_li=d1_lstm_li,
#         act_li_lstm1=act_li_lstm1, act_li_lstm2=act_li_lstm2, act_li_lstm3=act_li_lstm3, act_li_lstm4=act_li_lstm4,
#         do_lstm_li_1=do_lstm_li_1, do_lstm_li_2=do_lstm_li_2, do_lstm_li_3=do_lstm_li_3, do_lstm_li_4=do_lstm_li_4,
#         d1_fuse=d1_fuse, act_fuse=act_fuse, dropout_rate_fuse=dropout_rate_fuse,
#         d3_end=d3_end, d2_end=d2_end, 
#         act_end_1=act_end_1, act_end_2=act_end_2
#     )

#     # Compile the model
#     model.compile(loss=combined_loss, optimizer=optimizer, metrics=['mae'])
    
#     return model