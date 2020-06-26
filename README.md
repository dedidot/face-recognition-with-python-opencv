# Documentation
This is a way to check pictures of people with face_recognition.

![Result](https://github.com/dedidot/face-recognition-with-python-opencv/blob/master/Screen%20Shot%202020-06-26%20at%2016.06.44.png)

**Quict Start**
1. Clone repo, `git clone https://github.com/dedidot/face-recognition-with-python-opencv.git`
2. install  `requirements.txt`  using  `pip install -r requirements.txt`
3. Run `python index.py`

**Explanation**
Load known faces folder:

```
for name in os.listdir(KNOWN_FACES_DIR):
   if name ==  '.DS_Store':
      continue
   for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
       	image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
    	encoding = face_recognition.face_encodings(image)[0]
    	known_faces.append(encoding)
    	known_names.append(name)
```

Load unknown faces and compare with known faces:

```
for filename in os.listdir(UNKNOWN_FACES_DIR):
    print(f'Filename {filename}', end='')
    image = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}/{filename}')
    
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    print(f' found {len(encodings)} face(s) ')
    
    for face_encoding, face_location in  zip(encodings, locations):
    	results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
	match =  None
	
	if  True  in results:
	    match = known_names[results.index(True)]
	    
	    print(f' - {match} from {results} ')
	    
	    top_left = (face_location[3], face_location[0])
	    bottom_right = (face_location[1], face_location[2])
	    
	    color = name_to_color(match)
	    cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
	    
	    top_left = (face_location[3], face_location[2])
	    bottom_right = (face_location[1], face_location[2] +  22)
	    
	    cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
	    cv2.putText(image, match, (face_location[3] +  10, face_location[2] +  15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)
```

**Show result:**

    cv2.imshow(filename, image)
    cv2.waitKey(10000)
    cv2.destroyWindow(filename)

Inspired by: sentdex (Harrison)
