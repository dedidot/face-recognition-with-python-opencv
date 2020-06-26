import face_recognition
import os
import cv2

KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
TOLERANCE = 0.5
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "cnn" #hog

print("loading known faces")

known_faces = []
known_names = []

# Return (RGB) from name
def name_to_color(name):
    # Take 3 firts letters, to lower()
    # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color

for name in os.listdir(KNOWN_FACES_DIR):
    
    if name == '.DS_Store':
        continue
    #Load every file of faces of known person
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')

        # Get 128-dimention face encoding
        encoding = face_recognition.face_encodings(image)[0]

        # Append encoding and image name
        known_faces.append(encoding)
        known_names.append(name)

print("Processing unknown faces....")

for filename in os.listdir(UNKNOWN_FACES_DIR):

    #Load image
    print(f'Filename {filename}', end='')
    image = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}/{filename}')

    # Grab face location
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)

    # Convert RGB to BGR with cv2
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    print(f' found {len(encodings)} face(s) ')

    for face_encoding, face_location in zip(encodings, locations):

        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

        # Check unknown faces with known faces
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f' - {match} from {results} ')

            # Each location contains position in order: top, right, bottom, left
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            # Get color by name using our fancy function
            color = name_to_color(match)
            
            # Paint frame
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            # Now we need smaller, filled grame bellow for a name
            # This time we use bottom in both corners - to start from bottom and moce 50 pixels down
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)

            # Paint frame
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

            # Write a name
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

    # Show image
    cv2.imshow(filename, image)
    cv2.waitKey(10000)
    #cv2.destroyWindow(filename)