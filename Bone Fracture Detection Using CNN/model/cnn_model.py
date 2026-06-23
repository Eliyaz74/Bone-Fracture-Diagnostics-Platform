def build_cnn_model(input_shape=(224, 224, 3), num_classes=1):
    """
    Build a CNN model.
    - num_classes=1 -> binary classification
    - num_classes>1 -> multi-class classification
    """
    model = Sequential()

    # 1st Convolution Block
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 2nd Convolution Block
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 3rd Convolution Block
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten
    model.add(Flatten())

    # Fully Connected Layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Output Layer
    if num_classes == 1:
        model.add(Dense(1, activation='sigmoid'))  # Binary
        loss_fn = 'binary_crossentropy'
    else:
        model.add(Dense(num_classes, activation='softmax'))  # Multi-class
        loss_fn = 'categorical_crossentropy'

    # Compile Model
    model.compile(
        optimizer='adam',
        loss=loss_fn,
        metrics=['accuracy']
    )

    return model