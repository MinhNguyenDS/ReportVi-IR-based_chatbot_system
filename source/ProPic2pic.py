import os
import cv2
import numpy as np
import sqlite3
import pickle

PATH_IMAGE = []

def extract_and_save_feature(image_path, label, cursor, conn, model, preprocess_input):
    # Read and preprocess images
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = preprocess_input(image)

    # Get features from the model
    feature = model.predict(np.expand_dims(image, axis=0)).flatten()

    # Save to SQLite
    cursor.execute('INSERT INTO features (filepath, label, feature) VALUES (?, ?, ?)', (image_path, label, pickle.dumps(feature)))
    conn.commit()


def offline_process_pic2pic(dataset_path, imgs_bar, model, preprocess_input):
    PATH_IMAGE.append(dataset_path)
    if os.path.exists('vectorstores/sqlite3_pic2pic.db'): os.remove('vectorstores/sqlite3_pic2pic.db')
    # Connect to SQLite database
    conn = sqlite3.connect('vectorstores/sqlite3_pic2pic.db')
    cursor = conn.cursor()

    # Create a table to store features
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT NOT NULL,
            label TEXT NOT NULL,
            feature BLOB NOT NULL
        )
    ''')
    conn.commit()

    # Get and save features for all images in the dataset
    len_label = len(os.listdir(dataset_path))
    i=0
    for label in os.listdir(dataset_path):
        i += 1
        imgs_bar.progress((i/len_label)*0.8, text=f'{i}/{len_label} class: extracting and saving feature to vectorstore.')

        label_path = os.path.join(dataset_path, label)

        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                filepath = os.path.join(label_path, filename)
                extract_and_save_feature(filepath, label, cursor, conn, model, preprocess_input)

    # Close the connection to SQLite
    conn.close()

def calculate_cosine_similarity(feature1, feature2):
    dot_product = np.dot(feature1, feature2)
    norm_feature1 = np.linalg.norm(feature1)
    norm_feature2 = np.linalg.norm(feature2)
    similarity = dot_product / (norm_feature1 * norm_feature2)
    return similarity

def find_similar_image(query_feature, top_n=5):

    if PATH_IMAGE != []: conn = sqlite3.connect('vectorstores/sqlite3_pic2pic.db')
    else: conn = sqlite3.connect('vectorstores/sqlite3_pic2pic_full.db')
    cursor = conn.cursor()

    # Get characteristics of all images in the database
    cursor.execute('SELECT filepath, label, feature FROM features')
    rows = cursor.fetchall()

    # Calculate similarity and get top n similar images
    similarities = []
    for row in rows:
        filepath, label, saved_feature = row
        saved_feature = pickle.loads(saved_feature)
        similarity = calculate_cosine_similarity(query_feature, saved_feature)
        similarities.append((filepath, label, similarity))

    # Sort by similarity in descending order
    similarities.sort(key=lambda x: x[2], reverse=True)

    conn.close()
    return similarities[:top_n]
