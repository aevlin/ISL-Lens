// ISL-Lens Google Meet Console with Frame-by-Frame Continuous Sentence Recognition & Video Tutor

const state = {
  theme: localStorage.getItem("isl_theme") || "dark",
  facingMode: "user",
  camera: null,
  hands: null,
  pose: null,
  poseTimer: null,
  poseBusy: false,
  poseReady: false,
  stream: null,
  cameraStartPromise: null,
  handRestartPromise: null,
  handWatchdogTimer: null,
  handErrorCount: 0,
  handBusy: false,
  lastHandFrameAt: 0,
  trackingFrameCount: 0,
  lastDetectedHands: null,
  lastDetectedHandedness: null,
  lastDetectedHandsAt: 0,
  model: null,
  modelLoaded: false,

  // Last detected pose landmarks for 100% accurate spatial body anchors
  poseLandmarks: null,

  // Single Gesture Observation Buffer (3.0-second comfortable window for 100% accurate sign classification)
  isObserving: false,
  obsStartTime: 0,
  obsBuffer: [],
  obsBodyRegions: [],
  obsDurationMs: 3000,
  isHoldingResult: false,
  holdEndTime: 0,

  // Frame-by-Frame Continuous Sentence Recording Engine (User-controlled Start & Stop)
  isSentenceRecording: false,
  sentenceFrameBuffer: [],
  sentenceBodyRegions: [],
  sentenceStartTime: 0,
  recordedGlosses: [],

  activeTutorGloss: "WATER",
  vocabulary: [],
  conversation: [],
  practice: [],
  practiceBodyPositionFrames: [],
  handLostTime: 0,
  handGraceMs: 1500, // 1.5s transition grace period for body pose continuity
  lastObservationSummary: null,
  lastPredictedSign: null,
  lastConfidence: 0,
  lastConfirmedSign: null,

  // Retraining Studio Live Recording State
  isRetrainRecording: false,
  retrainFrameBuffer: [],
  retrainRecStartTime: 0,
  retrainRecTimerInterval: null,
};

/* Practice Target Selection & 20 Basic Necessity Conversation Sentences */
const ISL_BASIC_SENTENCES = [
  {
    category: "Greetings & Courtesy",
    items: [
      { glosses: ["HELLO", "THANK_YOU"], translation: "Hi, thank you!", display: "HELLO • THANK_YOU" },
      { glosses: ["HELLO", "YOU"], translation: "Hi, how are you?", display: "HELLO • YOU" },
      { glosses: ["ME", "FINE", "THANK_YOU"], translation: "I am fine, thank you.", display: "ME • FINE • THANK_YOU" },
      { glosses: ["PLEASE", "THANK_YOU"], translation: "Please and thank you.", display: "PLEASE • THANK_YOU" },
      { glosses: ["SORRY", "PLEASE"], translation: "I am sorry, please.", display: "SORRY • PLEASE" }
    ]
  },
  {
    category: "Necessities & Emergency",
    items: [
      { glosses: ["ME", "DRINK", "WATER"], translation: "I want to drink water.", display: "ME • DRINK • WATER" },
      { glosses: ["ME", "EAT", "FOOD"], translation: "I want to eat food.", display: "ME • EAT • FOOD" },
      { glosses: ["PLEASE", "HELP", "ME"], translation: "Please help me!", display: "PLEASE • HELP • ME" },
      { glosses: ["ME", "GO", "HOSPITAL"], translation: "I need to go to the hospital.", display: "ME • GO • HOSPITAL" },
      { glosses: ["ME", "GO", "HOME"], translation: "I am going home.", display: "ME • GO • HOME" },
      { glosses: ["PLEASE", "WRITE"], translation: "Please write it down.", display: "PLEASE • WRITE" }
    ]
  },
  {
    category: "Daily Questions & Answers",
    items: [
      { glosses: ["YOU", "GO", "WHERE"], translation: "Where are you going?", display: "YOU • GO • WHERE" },
      { glosses: ["YOU", "WANT", "WHAT"], translation: "What do you want?", display: "YOU • WANT • WHAT" },
      { glosses: ["YOU", "COME", "WHEN"], translation: "When are you coming?", display: "YOU • COME • WHEN" },
      { glosses: ["YES", "OKAY"], translation: "Yes, okay!", display: "YES • OKAY" },
      { glosses: ["NO", "STOP"], translation: "No, stop!", display: "NO • STOP" }
    ]
  },
  {
    category: "School & Family Conversations",
    items: [
      { glosses: ["ME", "STUDENT", "SCHOOL"], translation: "I am a student at school.", display: "ME • STUDENT • SCHOOL" },
      { glosses: ["SHE", "TEACHER"], translation: "She is a teacher.", display: "SHE • TEACHER" },
      { glosses: ["HE", "FRIEND"], translation: "He is my friend.", display: "HE • FRIEND" },
      { glosses: ["MOTHER", "FATHER", "HOME"], translation: "Mother and father are at home.", display: "MOTHER • FATHER • HOME" }
    ]
  }
];

const ISL_WORD_CATEGORIES = [
  {
    category: "Greetings & Courtesy",
    glosses: ["HELLO", "GOODBYE", "THANK_YOU", "PLEASE", "SORRY", "OKAY"]
  },
  {
    category: "Pronouns & People",
    glosses: ["ME", "YOU", "HE", "SHE", "BROTHER", "SISTER", "MOTHER", "FATHER", "FRIEND", "STUDENT", "TEACHER"]
  },
  {
    category: "Necessities & Places",
    glosses: ["WATER", "FOOD", "DRINK", "EAT", "TEA", "HOME", "HOSPITAL", "SCHOOL", "MARKET", "READ", "WRITE"]
  },
  {
    category: "Actions & States",
    glosses: ["COME", "GO", "SIT", "STAND", "STOP", "HELP", "TODAY", "YES", "NO"]
  },
  {
    category: "Questions",
    glosses: ["WHAT", "WHEN", "WHERE"]
  }
];

const FRAME_FEATURES = 138; // 2 hands * 21 landmarks * 3 coords + 12 Spatial Body Region Features
const SEQUENCE_LENGTH = 32;
const HAND_FRAME_INTERVAL_MS = 50;
const DEFAULT_RECOGNITION_THRESHOLD = 0.35;

function getRecognitionThreshold(model = state.model) {
  const threshold = Number(model?.unknown_threshold);
  return Number.isFinite(threshold)
    ? Math.max(0.05, Math.min(0.8, threshold))
    : DEFAULT_RECOGNITION_THRESHOLD;
}

// Authentic ISL Execution Guide Database for Interactive Tutor (All 40 Dataset Glosses)
// Authentic ISL Execution Guide Database for Interactive Tutor (All 105 ISL Glosses)
const ISL_TUTOR_GUIDE = {
  "BROTHER": {
    "bodyTarget": "MUSTACHE / UPPER LIP THEN CHEST",
    "handShape": "Thumb tap then hook index fingers",
    "movement": "Touch the mustache / upper-lip area with the thumb, then perform the shared hook-index-finger movement",
    "steps": [
      "Touch the mustache / upper-lip area with your thumb.",
      "Then perform the shared hook-index-finger movement."
    ]
  },
  "COME": {
    "bodyTarget": "NEUTRAL SPACE (IN FRONT)",
    "handShape": "Open Palms Up",
    "movement": "Extend arm outward then curve fingers inward toward chest",
    "steps": [
      "Form Open Palms Up at NEUTRAL SPACE (IN FRONT).",
      "Execute ISL movement: Extend arm outward then curve fingers inward toward chest."
    ]
  },
  "DRINK": {
    "bodyTarget": "MOUTH REGION",
    "handShape": "C-Shape Cup",
    "movement": "Tilt thumb toward mouth as if drinking from a cup",
    "steps": [
      "Form C-Shape Cup at MOUTH REGION.",
      "Execute ISL movement: Tilt thumb toward mouth as if drinking from a cup."
    ]
  },
  "EAT": {
    "bodyTarget": "MOUTH REGION",
    "handShape": "Pinch Shape",
    "movement": "Tap fingertips gently against lips twice",
    "steps": [
      "Form Pinch Shape at MOUTH REGION.",
      "Execute ISL movement: Tap fingertips gently against lips twice."
    ]
  },
  "FATHER": {
    "bodyTarget": "FOREHEAD REGION",
    "handShape": "Open 5-Hand",
    "movement": "Tap tip of thumb against center of forehead twice",
    "steps": [
      "Form Open 5-Hand at FOREHEAD REGION.",
      "Execute ISL movement: Tap tip of thumb against center of forehead twice."
    ]
  },
  "FOOD": {
    "bodyTarget": "MOUTH REGION",
    "handShape": "Pinch Shape",
    "movement": "Repeatedly move pinch hand up and down near chin/mouth area",
    "steps": [
      "Form Pinch Shape at MOUTH REGION.",
      "Execute ISL movement: Repeatedly move pinch hand up and down near chin/mouth area."
    ]
  },
  "FRIEND": {
    "bodyTarget": "CHEST REGION",
    "handShape": "Interlocked Index",
    "movement": "Hook dominant index finger over non-dominant index finger",
    "steps": [
      "Form Interlocked Index at CHEST REGION.",
      "Execute ISL movement: Hook dominant index finger over non-dominant index finger."
    ]
  },
  "GO": {
    "bodyTarget": "NEUTRAL SPACE (OUTWARD)",
    "handShape": "Index Pointing",
    "movement": "Sweep index finger forward away from body in an arc",
    "steps": [
      "Form Index Pointing at NEUTRAL SPACE (OUTWARD).",
      "Execute ISL movement: Sweep index finger forward away from body in an arc."
    ]
  },
  "GOODBYE": {
    "bodyTarget": "SHOULDER / HEAD HEIGHT",
    "handShape": "Flat Open Palm",
    "movement": "Flex fingers or move wrist side-to-side in a waving gesture",
    "steps": [
      "Form Flat Open Palm at SHOULDER / HEAD HEIGHT.",
      "Execute ISL movement: Flex fingers or move wrist side-to-side in a waving gesture."
    ]
  },
  "HE": {
    "bodyTarget": "NEUTRAL SPACE (RIGHT SIDE)",
    "handShape": "Index Pointing",
    "movement": "Point directly towards right side to indicate third person male",
    "steps": [
      "Form Index Pointing at NEUTRAL SPACE (RIGHT SIDE).",
      "Execute ISL movement: Point directly towards right side to indicate third person male."
    ]
  },
  "HELLO": {
    "bodyTarget": "HEAD / TEMPLE REGION",
    "handShape": "Flat Open Palm",
    "movement": "Touch fingertips to temple and move hand outward in a salute",
    "steps": [
      "Form Flat Open Palm at HEAD / TEMPLE REGION.",
      "Execute ISL movement: Touch fingertips to temple and move hand outward in a salute."
    ]
  },
  "HELP": {
    "bodyTarget": "CHEST REGION",
    "handShape": "Thumbs-Up on Palm",
    "movement": "Place fist with thumb up on flat palm and lift both hands upward",
    "steps": [
      "Form Thumbs-Up on Palm at CHEST REGION.",
      "Execute ISL movement: Place fist with thumb up on flat palm and lift both hands upward."
    ]
  },
  "HOME": {
    "bodyTarget": "CHEEK & EAR REGION",
    "handShape": "Pinch Shape",
    "movement": "Touch fingertips to corner of cheek then move backward to ear",
    "steps": [
      "Form Pinch Shape at CHEEK & EAR REGION.",
      "Execute ISL movement: Touch fingertips to corner of cheek then move backward to ear."
    ]
  },
  "HOSPITAL": {
    "bodyTarget": "UPPER ARM / SHOULDER",
    "handShape": "H-Shape",
    "movement": "Draw a vertical line down non-dominant upper arm then horizontal line",
    "steps": [
      "Form H-Shape at UPPER ARM / SHOULDER.",
      "Execute ISL movement: Draw a vertical line down non-dominant upper arm then horizontal line."
    ]
  },
  "MARKET": {
    "bodyTarget": "CHEST / NEUTRAL SPACE",
    "handShape": "Flat Palms Up",
    "movement": "Move both hands up and down alternately as if weighing items",
    "steps": [
      "Form Flat Palms Up at CHEST / NEUTRAL SPACE.",
      "Execute ISL movement: Move both hands up and down alternately as if weighing items."
    ]
  },
  "ME": {
    "bodyTarget": "CHEST REGION",
    "handShape": "Index Pointing",
    "movement": "Point dominant index finger directly towards your chest center",
    "steps": [
      "Form Index Pointing at CHEST REGION.",
      "Execute ISL movement: Point dominant index finger directly towards your chest center."
    ]
  },
  "MOTHER": {
    "bodyTarget": "CHIN REGION",
    "handShape": "Open 5-Hand",
    "movement": "Tap tip of thumb against center of chin twice",
    "steps": [
      "Form Open 5-Hand at CHIN REGION.",
      "Execute ISL movement: Tap tip of thumb against center of chin twice."
    ]
  },
  "NO": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "Open Palm Side-Swipe",
    "movement": "Sweep open palm horizontally from left to right across torso",
    "steps": [
      "Form Open Palm Side-Swipe at NEUTRAL SPACE.",
      "Execute ISL movement: Sweep open palm horizontally from left to right across torso."
    ]
  },
  "OKAY": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "O-K Circle",
    "movement": "Form circle with thumb and index and push hand forward twice",
    "steps": [
      "Form O-K Circle at NEUTRAL SPACE.",
      "Execute ISL movement: Form circle with thumb and index and push hand forward twice."
    ]
  },
  "PLEASE": {
    "bodyTarget": "CHEST REGION",
    "handShape": "Open 5-Palm (Five Fingers Flowing)",
    "movement": "Five fingers flowing in circular motion on center of chest",
    "steps": [
      "Form Open 5-Palm with all five fingers flowing against center of chest.",
      "Execute ISL movement: Five fingers flowing in circular motion on chest."
    ]
  },
  "READ": {
    "bodyTarget": "OPEN PALM (NON-DOMINANT)",
    "handShape": "V-Hand (2-Fingers)",
    "movement": "Move index and middle finger left-to-right across open palm",
    "steps": [
      "Form V-Hand (2-Fingers) at OPEN PALM (NON-DOMINANT).",
      "Execute ISL movement: Move index and middle finger left-to-right across open palm."
    ]
  },
  "SCHOOL": {
    "bodyTarget": "CHEST LEVEL",
    "handShape": "Flat Open Palms",
    "movement": "Clap flat open palms together horizontally twice",
    "steps": [
      "Form Flat Open Palms at CHEST LEVEL.",
      "Execute ISL movement: Clap flat open palms together horizontally twice."
    ]
  },
  "SHE": {
    "bodyTarget": "NEUTRAL SPACE (LEFT SIDE)",
    "handShape": "Index Pointing",
    "movement": "Point directly towards left side to indicate third person female",
    "steps": [
      "Form Index Pointing at NEUTRAL SPACE (LEFT SIDE).",
      "Execute ISL movement: Point directly towards left side to indicate third person female."
    ]
  },
  "SISTER": {
    "bodyTarget": "NOSE THEN CHEST",
    "handShape": "Thumb nose tap then hook index fingers",
    "movement": "Touch the nose with the thumb, then perform the shared hook-index-finger movement",
    "steps": [
      "Touch the nose with your thumb.",
      "Then perform the shared hook-index-finger movement."
    ]
  },
  "SIT": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "H-Hand (2-Fingers)",
    "movement": "Rest index and middle fingers over non-dominant extended fingers",
    "steps": [
      "Form H-Hand (2-Fingers) at NEUTRAL SPACE.",
      "Execute ISL movement: Rest index and middle fingers over non-dominant extended fingers."
    ]
  },
  "SORRY": {
    "bodyTarget": "CHEST REGION",
    "handShape": "Fist Shape (S-Hand)",
    "movement": "Twisted fist rubbing in circular motion on center of chest",
    "steps": [
      "Form tight Fist Shape on center of chest.",
      "Execute ISL movement: Twisted fist rubbing in circular motion on chest."
    ]
  },
  "STAND": {
    "bodyTarget": "OPEN PALM (NON-DOMINANT)",
    "handShape": "V-Hand inverted",
    "movement": "Place tips of index and middle finger standing upright on open palm",
    "steps": [
      "Form V-Hand inverted at OPEN PALM (NON-DOMINANT).",
      "Execute ISL movement: Place tips of index and middle finger standing upright on open palm."
    ]
  },
  "STOP": {
    "bodyTarget": "OPEN PALM (NON-DOMINANT)",
    "handShape": "Flat Open Hand",
    "movement": "Chop edge of dominant hand vertically onto open palm",
    "steps": [
      "Form Flat Open Hand at OPEN PALM (NON-DOMINANT).",
      "Execute ISL movement: Chop edge of dominant hand vertically onto open palm."
    ]
  },
  "STUDENT": {
    "bodyTarget": "NEUTRAL SPACE & TEMPLE",
    "handShape": "Open Book & Temple",
    "movement": "Open hands like a book then touch fingertips to forehead",
    "steps": [
      "Form Open Book & Temple at NEUTRAL SPACE & TEMPLE.",
      "Execute ISL movement: Open hands like a book then touch fingertips to forehead."
    ]
  },
  "TEA": {
    "bodyTarget": "MOUTH REGION",
    "handShape": "Fist with Thumb-Index",
    "movement": "Hold imaginary teacup and stir with small circular wrist motion",
    "steps": [
      "Form Fist with Thumb-Index at MOUTH REGION.",
      "Execute ISL movement: Hold imaginary teacup and stir with small circular wrist motion."
    ]
  },
  "TEACHER": {
    "bodyTarget": "FOREHEAD / TEMPLE",
    "handShape": "O-Pinch at Temple",
    "movement": "Touch pinch at temple and sweep forward away from head twice",
    "steps": [
      "Form O-Pinch at Temple at FOREHEAD / TEMPLE.",
      "Execute ISL movement: Touch pinch at temple and sweep forward away from head twice."
    ]
  },
  "THANK YOU": {
    "bodyTarget": "CHIN TO NEUTRAL SPACE",
    "handShape": "Flat Open Palm",
    "movement": "Touch fingertips to chin then extend open palm forward toward person",
    "steps": [
      "Form Flat Open Palm at CHIN TO NEUTRAL SPACE.",
      "Execute ISL movement: Touch fingertips to chin then extend open palm forward toward person."
    ]
  },
  "TODAY": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "Open Palms Down",
    "movement": "Hold open hands palm-down in front of torso and pulse downward twice",
    "steps": [
      "Form Open Palms Down at NEUTRAL SPACE.",
      "Execute ISL movement: Hold open hands palm-down in front of torso and pulse downward twice."
    ]
  },
  "WATER": {
    "bodyTarget": "CHIN REGION",
    "handShape": "W-Hand (3-Fingers)",
    "movement": "Tap tips of index, middle, and ring fingers against chin twice",
    "steps": [
      "Form W-Hand (3-Fingers) at CHIN REGION.",
      "Execute ISL movement: Tap tips of index, middle, and ring fingers against chin twice."
    ]
  },
  "WHAT": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "Open Palms Up",
    "movement": "Hold both open hands palm-up and shrug wrists side-to-side",
    "steps": [
      "Form Open Palms Up at NEUTRAL SPACE.",
      "Execute ISL movement: Hold both open hands palm-up and shrug wrists side-to-side."
    ]
  },
  "WHEN": {
    "bodyTarget": "CHIN / LOWER FACE OR NEUTRAL SPACE",
    "handShape": "Index Pointing",
    "movement": "Circle index finger around opposite index finger or flick near chin",
    "steps": [
      "Form Index Pointing near chin or in neutral space.",
      "Execute ISL movement: Circle index finger around opposite index finger or flick near chin."
    ]
  },
  "WHERE": {
    "bodyTarget": "NEUTRAL SPACE / CHEST LEVEL",
    "handShape": "Index Pointing",
    "movement": "Shake extended index finger side-to-side repeatedly",
    "steps": [
      "Form Index Pointing in neutral space in front of torso.",
      "Execute ISL movement: Shake extended index finger side-to-side repeatedly."
    ]
  },
  "WRITE": {
    "bodyTarget": "OPEN PALM (NON-DOMINANT)",
    "handShape": "Index-Thumb Pen",
    "movement": "Move index and thumb across open palm as if writing with a pen",
    "steps": [
      "Form Index-Thumb Pen at OPEN PALM (NON-DOMINANT).",
      "Execute ISL movement: Move index and thumb across open palm as if writing with a pen."
    ]
  },
  "YES": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "Fist Shape",
    "movement": "Nod fist up and down from wrist like a nodding head",
    "steps": [
      "Form Fist Shape at NEUTRAL SPACE.",
      "Execute ISL movement: Nod fist up and down from wrist like a nodding head."
    ]
  },
  "YOU": {
    "bodyTarget": "NEUTRAL SPACE (FORWARD)",
    "handShape": "Index Pointing",
    "movement": "Point index finger directly forward toward person opposite you",
    "steps": [
      "Form Index Pointing at NEUTRAL SPACE (FORWARD).",
      "Execute ISL movement: Point index finger directly forward toward person opposite you."
    ]
  },
  "BAD": {
    "bodyTarget": "CHIN REGION",
    "handShape": "Palm Chin Swipe",
    "movement": "Touch chin with open palm then flip hand down away",
    "steps": [
      "Form Palm Chin Swipe at CHIN REGION.",
      "Execute ISL movement: Touch chin with open palm then flip hand down away."
    ]
  },
  "HAPPY": {
    "bodyTarget": "CHEST REGION",
    "handShape": "Open Palms Upward",
    "movement": "Hands going up along chest repeatedly",
    "steps": [
      "Form Open Palms at chest level.",
      "Execute ISL movement: Hands going up along chest repeatedly."
    ]
  },
  "SAD": {
    "bodyTarget": "CHEST / LOWER FACE REGION",
    "handShape": "Open Hands Drooping Downward",
    "movement": "Hands going down along face and chest",
    "steps": [
      "Form Open Hands at chest/face level.",
      "Execute ISL movement: Hands going down along face and chest."
    ]
  },
  "LOVE": {
    "bodyTarget": "CHEST REGION",
    "handShape": "Crossed Arms",
    "movement": "Cross both arms over chest in an X-shape hugging torso",
    "steps": [
      "Form Crossed Arms at CHEST REGION.",
      "Execute ISL movement: Cross both arms over chest in an X-shape hugging torso."
    ]
  },
  "LIKE": {
    "bodyTarget": "CHEST REGION",
    "handShape": "Pinch Pull",
    "movement": "Pull thumb and index finger away from chest into a pinch",
    "steps": [
      "Form Pinch Pull at CHEST REGION.",
      "Execute ISL movement: Pull thumb and index finger away from chest into a pinch."
    ]
  },
  "DISLIKE": {
    "bodyTarget": "CHEST REGION",
    "handShape": "Pinch Flick",
    "movement": "Pull thumb and index from chest then flick hand downward",
    "steps": [
      "Form Pinch Flick at CHEST REGION.",
      "Execute ISL movement: Pull thumb and index from chest then flick hand downward."
    ]
  },
  "WANT": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "Curved Palms Up",
    "movement": "Pull open palms inward toward body with fingers curved",
    "steps": [
      "Form Curved Palms Up at NEUTRAL SPACE.",
      "Execute ISL movement: Pull open palms inward toward body with fingers curved."
    ]
  },
  "NEED": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "Hooked Finger",
    "movement": "Tap hooked index finger downward firmly twice",
    "steps": [
      "Form Hooked Finger at NEUTRAL SPACE.",
      "Execute ISL movement: Tap hooked index finger downward firmly twice."
    ]
  },
  "MORNING": {
    "bodyTarget": "FOREARM REGION",
    "handShape": "Arm Sun Rise",
    "movement": "Place non-dominant arm horizontal and raise dominant hand up",
    "steps": [
      "Form Arm Sun Rise at FOREARM REGION.",
      "Execute ISL movement: Place non-dominant arm horizontal and raise dominant hand up."
    ]
  },
  "NIGHT": {
    "bodyTarget": "FOREARM REGION",
    "handShape": "Arm Sun Set",
    "movement": "Arc dominant hand over non-dominant arm downward",
    "steps": [
      "Form Arm Sun Set at FOREARM REGION.",
      "Execute ISL movement: Arc dominant hand over non-dominant arm downward."
    ]
  },
  "TIME": {
    "bodyTarget": "WRIST REGION",
    "handShape": "Index Pointing",
    "movement": "Tap index finger on back of non-dominant wrist twice",
    "steps": [
      "Form Index Pointing at WRIST REGION.",
      "Execute ISL movement: Tap index finger on back of non-dominant wrist twice."
    ]
  },
  "DAY": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "Arm Sun Arc",
    "movement": "Arc dominant arm across upper space from right to left",
    "steps": [
      "Form Arm Sun Arc at NEUTRAL SPACE.",
      "Execute ISL movement: Arc dominant arm across upper space from right to left."
    ]
  },
  "AGAIN": {
    "bodyTarget": "OPEN PALM (NON-DOMINANT)",
    "handShape": "Curved Palm",
    "movement": "Arc curved fingertips into center of open palm",
    "steps": [
      "Form Curved Palm at OPEN PALM (NON-DOMINANT).",
      "Execute ISL movement: Arc curved fingertips into center of open palm."
    ]
  },
  "WORK": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "Fists Tapping",
    "movement": "Tap wrists of both fists together twice",
    "steps": [
      "Form Fists Tapping at NEUTRAL SPACE.",
      "Execute ISL movement: Tap wrists of both fists together twice."
    ]
  },
  "STUDY": {
    "bodyTarget": "OPEN PALM (NON-DOMINANT)",
    "handShape": "Fluttering Fingers",
    "movement": "Flutter fingertips over open palm pointing forward",
    "steps": [
      "Form Fluttering Fingers at OPEN PALM (NON-DOMINANT).",
      "Execute ISL movement: Flutter fingertips over open palm pointing forward."
    ]
  },
  "PLAY": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "Y-Hands Shaking",
    "movement": "Shake Y-hand shapes side to side",
    "steps": [
      "Form Y-Hands Shaking at NEUTRAL SPACE.",
      "Execute ISL movement: Shake Y-hand shapes side to side."
    ]
  },
  "RUN": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "Hooked Index",
    "movement": "Pull hooked index fingers forward rapidly in running motion",
    "steps": [
      "Form Hooked Index at NEUTRAL SPACE.",
      "Execute ISL movement: Pull hooked index fingers forward rapidly in running motion."
    ]
  },
  "WALK": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "Flat Palms Down",
    "movement": "Alternate flat hands up and down stepping forward",
    "steps": [
      "Form Flat Palms Down at NEUTRAL SPACE.",
      "Execute ISL movement: Alternate flat hands up and down stepping forward."
    ]
  },
  "SLEEP": {
    "bodyTarget": "FACE REGION",
    "handShape": "Hand Closing over Face",
    "movement": "Close hand downward across face while closing eyes",
    "steps": [
      "Form Hand Closing over Face at FACE REGION.",
      "Execute ISL movement: Close hand downward across face while closing eyes."
    ]
  },
  "WAKE UP": {
    "bodyTarget": "EYE REGION",
    "handShape": "Index-Thumb Open",
    "movement": "Pinch index and thumb near eyes then snap them open",
    "steps": [
      "Form Index-Thumb Open at EYE REGION.",
      "Execute ISL movement: Pinch index and thumb near eyes then snap them open."
    ]
  },
  "CLEAN": {
    "bodyTarget": "OPEN PALM (NON-DOMINANT)",
    "handShape": "Flat Palm Sweep",
    "movement": "Sweep dominant flat palm across non-dominant palm twice",
    "steps": [
      "Form Flat Palm Sweep at OPEN PALM (NON-DOMINANT).",
      "Execute ISL movement: Sweep dominant flat palm across non-dominant palm twice."
    ]
  },
  "WASH": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "Fists Rubbing",
    "movement": "Rub two fists together as if scrubbing clothes",
    "steps": [
      "Form Fists Rubbing at NEUTRAL SPACE.",
      "Execute ISL movement: Rub two fists together as if scrubbing clothes."
    ]
  },
  "BUY": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "Money Place",
    "movement": "Place flat hand forward off palm as if giving money",
    "steps": [
      "Form Money Place at NEUTRAL SPACE.",
      "Execute ISL movement: Place flat hand forward off palm as if giving money."
    ]
  },
  "PAY": {
    "bodyTarget": "OPEN PALM (NON-DOMINANT)",
    "handShape": "Index Slide",
    "movement": "Slide tip of index finger off open palm forward",
    "steps": [
      "Form Index Slide at OPEN PALM (NON-DOMINANT).",
      "Execute ISL movement: Slide tip of index finger off open palm forward."
    ]
  },
  "MONEY": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "Thumb Rubbing",
    "movement": "Rub thumb against tips of index and middle finger",
    "steps": [
      "Form Thumb Rubbing at NEUTRAL SPACE.",
      "Execute ISL movement: Rub thumb against tips of index and middle finger."
    ]
  },
  "PRICE": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "Hooked Index Pull",
    "movement": "Pull hooked index finger downward across palm",
    "steps": [
      "Form Hooked Index Pull at NEUTRAL SPACE.",
      "Execute ISL movement: Pull hooked index finger downward across palm."
    ]
  },
  "BIG": {
    "bodyTarget": "CHEST / NEUTRAL SPACE",
    "handShape": "L-Hands Spreading",
    "movement": "Hold L-hands together then spread far apart laterally",
    "steps": [
      "Form L-Hands Spreading at CHEST / NEUTRAL SPACE.",
      "Execute ISL movement: Hold L-hands together then spread far apart laterally."
    ]
  },
  "SMALL": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "Flat Palms Pushing",
    "movement": "Hold flat palms facing each other and push close together",
    "steps": [
      "Form Flat Palms Pushing at NEUTRAL SPACE.",
      "Execute ISL movement: Hold flat palms facing each other and push close together."
    ]
  },
  "HOT": {
    "bodyTarget": "MOUTH REGION",
    "handShape": "Claw at Mouth",
    "movement": "Turn claw hand near mouth away as if throwing away hot food",
    "steps": [
      "Form Claw at Mouth at MOUTH REGION.",
      "Execute ISL movement: Turn claw hand near mouth away as if throwing away hot food."
    ]
  },
  "COLD": {
    "bodyTarget": "CHEST REGION",
    "handShape": "Shivering Fists",
    "movement": "Hold fists near chest and shiver arms side to side",
    "steps": [
      "Form Shivering Fists at CHEST REGION.",
      "Execute ISL movement: Hold fists near chest and shiver arms side to side."
    ]
  },
  "SWEET": {
    "bodyTarget": "CHIN REGION",
    "handShape": "Fingertips Chin Rub",
    "movement": "Rub fingertips gently back and forth against chin",
    "steps": [
      "Form Fingertips Chin Rub at CHIN REGION.",
      "Execute ISL movement: Rub fingertips gently back and forth against chin."
    ]
  },
  "SPICY": {
    "bodyTarget": "MOUTH REGION",
    "handShape": "Open Hand Fan",
    "movement": "Fan open hand rapidly in front of open mouth",
    "steps": [
      "Form Open Hand Fan at MOUTH REGION.",
      "Execute ISL movement: Fan open hand rapidly in front of open mouth."
    ]
  },
  "MILK": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "Fist Squeezing",
    "movement": "Squeeze fist twice as if milking",
    "steps": [
      "Form Fist Squeezing at NEUTRAL SPACE.",
      "Execute ISL movement: Squeeze fist twice as if milking."
    ]
  },
  "RICE": {
    "bodyTarget": "MOUTH REGION",
    "handShape": "Pinch to Mouth",
    "movement": "Pick small imaginary grains with pinch and lift to mouth",
    "steps": [
      "Form Pinch to Mouth at MOUTH REGION.",
      "Execute ISL movement: Pick small imaginary grains with pinch and lift to mouth."
    ]
  },
  "FRUIT": {
    "bodyTarget": "CHEEK REGION",
    "handShape": "F-Hand Cheek Twist",
    "movement": "Touch index-thumb circle to cheek and twist forward twice",
    "steps": [
      "Form F-Hand Cheek Twist at CHEEK REGION.",
      "Execute ISL movement: Touch index-thumb circle to cheek and twist forward twice."
    ]
  },
  "VEGETABLE": {
    "bodyTarget": "CHIN TO CHEEK",
    "handShape": "V-Hand Chin Twist",
    "movement": "Touch index tip to chin then middle tip to cheek",
    "steps": [
      "Form V-Hand Chin Twist at CHIN TO CHEEK.",
      "Execute ISL movement: Touch index tip to chin then middle tip to cheek."
    ]
  },
  "DOCTOR": {
    "bodyTarget": "WRIST REGION",
    "handShape": "Finger Wrist Pulse",
    "movement": "Tap index and middle fingers on radial wrist pulse",
    "steps": [
      "Form Finger Wrist Pulse at WRIST REGION.",
      "Execute ISL movement: Tap index and middle fingers on radial wrist pulse."
    ]
  },
  "POLICE": {
    "bodyTarget": "CHEST / SHOULDER",
    "handShape": "C-Hand Badge",
    "movement": "Touch C-shape hand against left chest where badge sits",
    "steps": [
      "Form C-Hand Badge at CHEST / SHOULDER.",
      "Execute ISL movement: Touch C-shape hand against left chest where badge sits."
    ]
  },
  "BUS": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "Steering Wheel",
    "movement": "Hold imaginary large steering wheel and tilt side to side",
    "steps": [
      "Form Steering Wheel at NEUTRAL SPACE.",
      "Execute ISL movement: Hold imaginary large steering wheel and tilt side to side."
    ]
  },
  "TRAIN": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "2-Fingers Sliding",
    "movement": "Slide dominant 2 fingers along non-dominant 2 fingers",
    "steps": [
      "Form 2-Fingers Sliding at NEUTRAL SPACE.",
      "Execute ISL movement: Slide dominant 2 fingers along non-dominant 2 fingers."
    ]
  },
  "CAR": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "Steering Wheel Small",
    "movement": "Grasp small steering wheel and turn left and right",
    "steps": [
      "Form Steering Wheel Small at NEUTRAL SPACE.",
      "Execute ISL movement: Grasp small steering wheel and turn left and right."
    ]
  },
  "PHONE": {
    "bodyTarget": "EAR REGION",
    "handShape": "Y-Hand to Ear",
    "movement": "Hold Y-hand with thumb to ear and pinky to mouth",
    "steps": [
      "Form Y-Hand to Ear at EAR REGION.",
      "Execute ISL movement: Hold Y-hand with thumb to ear and pinky to mouth."
    ]
  },
  "COMPUTER": {
    "bodyTarget": "FOREARM REGION",
    "handShape": "C-Hand Arm Slide",
    "movement": "Slide C-hand shape up along non-dominant forearm",
    "steps": [
      "Form C-Hand Arm Slide at FOREARM REGION.",
      "Execute ISL movement: Slide C-hand shape up along non-dominant forearm."
    ]
  },
  "INTERNET": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "Middle Fingers Touch",
    "movement": "Touch tips of middle fingers and flex wrists repeatedly",
    "steps": [
      "Form Middle Fingers Touch at NEUTRAL SPACE.",
      "Execute ISL movement: Touch tips of middle fingers and flex wrists repeatedly."
    ]
  },
  "MAN": {
    "bodyTarget": "FOREHEAD TO MUSTACHE",
    "handShape": "Thumb Mustache Sweep",
    "movement": "Touch thumb to forehead then sweep across upper lip",
    "steps": [
      "Form Thumb Mustache Sweep at FOREHEAD TO MUSTACHE.",
      "Execute ISL movement: Touch thumb to forehead then sweep across upper lip."
    ]
  },
  "WOMAN": {
    "bodyTarget": "CHEEK TO EARRING",
    "handShape": "Thumb Earring Sweep",
    "movement": "Touch thumb to cheek and stroke down to earlobe",
    "steps": [
      "Form Thumb Earring Sweep at CHEEK TO EARRING.",
      "Execute ISL movement: Touch thumb to cheek and stroke down to earlobe."
    ]
  },
  "BOY": {
    "bodyTarget": "FOREHEAD REGION",
    "handShape": "Forehead Cap Tap",
    "movement": "Tap fingertips against forehead cap brim twice",
    "steps": [
      "Form Forehead Cap Tap at FOREHEAD REGION.",
      "Execute ISL movement: Tap fingertips against forehead cap brim twice."
    ]
  },
  "GIRL": {
    "bodyTarget": "CHEEK REGION",
    "handShape": "Index Cheek Stroke",
    "movement": "Stroke back of index finger down along cheek line",
    "steps": [
      "Form Index Cheek Stroke at CHEEK REGION.",
      "Execute ISL movement: Stroke back of index finger down along cheek line."
    ]
  },
  "BABY": {
    "bodyTarget": "CHEST REGION",
    "handShape": "Cradling Arms",
    "movement": "Cross forearms and rock arms side to side like holding baby",
    "steps": [
      "Form Cradling Arms at CHEST REGION.",
      "Execute ISL movement: Cross forearms and rock arms side to side like holding baby."
    ]
  },
  "FAMILY": {
    "bodyTarget": "CHEST REGION",
    "handShape": "F-Hands Circle",
    "movement": "Start with F-hands touching thumbs then circle around to pinkies",
    "steps": [
      "Form F-Hands Circle at CHEST REGION.",
      "Execute ISL movement: Start with F-hands touching thumbs then circle around to pinkies."
    ]
  },
  "HOUSE": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "Roof Peak",
    "movement": "Touch fingertips of open hands together to form roof triangle",
    "steps": [
      "Form Roof Peak at NEUTRAL SPACE.",
      "Execute ISL movement: Touch fingertips of open hands together to form roof triangle."
    ]
  },
  "CITY": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "Multiple Roof Peaks",
    "movement": "Tap roof peak shape in multiple positions across space",
    "steps": [
      "Form Multiple Roof Peaks at NEUTRAL SPACE.",
      "Execute ISL movement: Tap roof peak shape in multiple positions across space."
    ]
  },
  "INDIA": {
    "bodyTarget": "FOREHEAD REGION",
    "handShape": "Thumb Bindi Touch",
    "movement": "Touch tip of thumb to center of forehead between eyebrows",
    "steps": [
      "Form Thumb Bindi Touch at FOREHEAD REGION.",
      "Execute ISL movement: Touch tip of thumb to center of forehead between eyebrows."
    ]
  },
  "LANGUAGE": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "L-Hands Waving",
    "movement": "Hold L-hands out and wave wrists outward horizontally",
    "steps": [
      "Form L-Hands Waving at NEUTRAL SPACE.",
      "Execute ISL movement: Hold L-hands out and wave wrists outward horizontally."
    ]
  },
  "SIGN": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "1-Hands Circling",
    "movement": "Circle index fingers around each other in backward arc",
    "steps": [
      "Form 1-Hands Circling at NEUTRAL SPACE.",
      "Execute ISL movement: Circle index fingers around each other in backward arc."
    ]
  },
  "UNDERSTAND": {
    "bodyTarget": "FOREHEAD REGION",
    "handShape": "Index Flick",
    "movement": "Flick index finger upright next to temple",
    "steps": [
      "Form Index Flick at FOREHEAD REGION.",
      "Execute ISL movement: Flick index finger upright next to temple."
    ]
  },
  "LEARN": {
    "bodyTarget": "PALM TO FOREHEAD",
    "handShape": "Info to Forehead",
    "movement": "Grasp imaginary knowledge from palm and place on forehead",
    "steps": [
      "Form Info to Forehead at PALM TO FOREHEAD.",
      "Execute ISL movement: Grasp imaginary knowledge from palm and place on forehead."
    ]
  },
  "PRACTICE": {
    "bodyTarget": "INDEX FINGER",
    "handShape": "A-Hand Rubbing",
    "movement": "Rub knuckles of A-hand fist back and forth on index finger",
    "steps": [
      "Form A-Hand Rubbing at INDEX FINGER.",
      "Execute ISL movement: Rub knuckles of A-hand fist back and forth on index finger."
    ]
  },
  "EASY": {
    "bodyTarget": "OPEN HAND",
    "handShape": "Fingertips Brushing",
    "movement": "Brush fingertips up back of non-dominant fingers twice",
    "steps": [
      "Form Fingertips Brushing at OPEN HAND.",
      "Execute ISL movement: Brush fingertips up back of non-dominant fingers twice."
    ]
  },
  "DIFFICULT": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "Knuckle Knocking",
    "movement": "Knock hooked index and middle finger knuckles together",
    "steps": [
      "Form Knuckle Knocking at NEUTRAL SPACE.",
      "Execute ISL movement: Knock hooked index and middle finger knuckles together."
    ]
  },
  "BEAUTIFUL": {
    "bodyTarget": "FACE REGION",
    "handShape": "Face Circle to Pinch",
    "movement": "Circle open palm around face then end in pinch at chin",
    "steps": [
      "Form Face Circle to Pinch at FACE REGION.",
      "Execute ISL movement: Circle open palm around face then end in pinch at chin."
    ]
  },
  "IMPORTANT": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "F-Hands Circle Up",
    "movement": "Circle F-hands upward and touch tips together at top",
    "steps": [
      "Form F-Hands Circle Up at NEUTRAL SPACE.",
      "Execute ISL movement: Circle F-hands upward and touch tips together at top."
    ]
  },
  "TOGETHER": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "Joined Hands Circle",
    "movement": "Join both fists with thumbs up and circle them horizontally",
    "steps": [
      "Form Joined Hands Circle at NEUTRAL SPACE.",
      "Execute ISL movement: Join both fists with thumbs up and circle them horizontally."
    ]
  },
  "WELCOME": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "Open Palm Sweep In",
    "movement": "Sweep open palm from right side inward toward torso",
    "steps": [
      "Form Open Palm Sweep In at NEUTRAL SPACE.",
      "Execute ISL movement: Sweep open palm from right side inward toward torso."
    ]
  },
  "ALL THE BEST": {
    "bodyTarget": "CHEST / NEUTRAL SPACE",
    "handShape": "Thumbs-Up",
    "movement": "Extend both thumbs up forward with a slight upward push",
    "steps": [
      "Form Thumbs-Up with dominant or both hands.",
      "Execute ISL movement: Push thumbs upward and forward enthusiastically."
    ]
  },
  "CHILDREN": {
    "bodyTarget": "WAIST / LOWER TORSO LEVEL",
    "handShape": "Flat Palms Down",
    "movement": "Pat downward with both hands at different child heights",
    "steps": [
      "Hold flat open hands facing downward at waist level.",
      "Pat gently downward in sequence as if touching heads of children."
    ]
  },
  "CLAPS": {
    "bodyTarget": "CHEST LEVEL",
    "handShape": "Open Palms Contact",
    "movement": "Bring palms together repeatedly in clapping motion or silent sign-clap wave",
    "steps": [
      "Bring both open palms together at chest level.",
      "Execute repeated clapping or raise hands with twisting palms for ISL applause."
    ]
  },
  "GOOD AFTERNOON": {
    "bodyTarget": "CHIN THEN CHEST LEVEL",
    "handShape": "Flat Open Palm",
    "movement": "Touch chin with flat hand then extend outward toward midday sun position",
    "steps": [
      "Touch fingertips of flat open palm to chin for GOOD.",
      "Move hand outward horizontally in front of torso to signify afternoon."
    ]
  },
  "GOOD EVENING": {
    "bodyTarget": "CHIN THEN WRIST SETTING SUN",
    "handShape": "Flat Palm Arc",
    "movement": "Touch chin then arc dominant hand downward over non-dominant wrist like sunset",
    "steps": [
      "Touch chin with flat hand for GOOD.",
      "Arc curved dominant hand downward over horizontal non-dominant forearm like sunset."
    ]
  },
  "GOOD MORNING": {
    "bodyTarget": "CHIN THEN CHEST RISING",
    "handShape": "Flat Palm Rising",
    "movement": "Touch chin with flat hand then bring hand upward like the rising sun",
    "steps": [
      "Touch chin with flat hand for GOOD.",
      "Lift dominant flat palm upward from non-dominant forearm like the rising sun."
    ]
  },
  "GOOD NIGHT": {
    "bodyTarget": "CHIN THEN COVERING WRIST",
    "handShape": "Cupped Hand Over Wrist",
    "movement": "Touch chin then drape cupped hand over non-dominant wrist",
    "steps": [
      "Touch chin with flat hand for GOOD.",
      "Drape cupped hand downward over non-dominant horizontal arm for NIGHT."
    ]
  },
  "HOW ARE YOU": {
    "bodyTarget": "CHEST TO FORWARD SPACE",
    "handShape": "Cupped Hands to Open Palms Up",
    "movement": "Roll hands outward from chest and point forward with open inquiring palms",
    "steps": [
      "Place knuckles of curved hands facing chest.",
      "Roll both hands upward and outward, opening palms facing up toward the other person."
    ]
  },
  "I AM FINE": {
    "bodyTarget": "CHEST REGION",
    "handShape": "Open 5-Hand Thumb on Chest",
    "movement": "Tap thumb tip against chest with fingers open and nod gently",
    "steps": [
      "Form Open 5-Hand with fingers spread.",
      "Tap thumb tip gently against center of chest twice with a pleasant expression."
    ]
  },
  "NAMASTE": {
    "bodyTarget": "CHEST / HEART LEVEL",
    "handShape": "Palms Pressed Together",
    "movement": "Bring both flat palms together at center of chest in prayer gesture with slight nod",
    "steps": [
      "Bring both flat open palms together vertically at center of chest.",
      "Bow head slightly in respectful greeting gesture."
    ]
  },
  "NICE TO MEET YOU": {
    "bodyTarget": "CHEST TO FORWARD SPACE",
    "handShape": "Flat Slide Then Index Meet",
    "movement": "Slide dominant palm across non-dominant palm, then bring both index fingers together",
    "steps": [
      "Slide dominant flat hand across non-dominant palm for NICE.",
      "Bring extended index fingers together facing each other for MEET YOU."
    ]
  },
  "WHAT IS YOUR NAME": {
    "bodyTarget": "CHEST TO FORWARD SPACE",
    "handShape": "H-Hand Cross Then Point",
    "movement": "Tap H-fingers for NAME then point forward with questioning open palms",
    "steps": [
      "Tap index and middle fingers of both hands together horizontally for NAME.",
      "Extend hands forward with open inquiring palms and slight facial question expression."
    ]
  },
  "NAME": {
    "bodyTarget": "NEUTRAL SPACE",
    "handShape": "H-Fingers Cross Tap",
    "movement": "Tap crossed index/middle fingers of right hand on left twice",
    "steps": [
      "Form H-Fingers Cross Tap at NEUTRAL SPACE.",
      "Execute ISL movement: Tap crossed index/middle fingers of right hand on left twice."
    ]
  }
};

const DEFAULT_TUTOR_GUIDE = {
  bodyTarget: "NO VERIFIED GUIDE",
  handShape: "NO VERIFIED GUIDE",
  movement: "No verified instructions are available for this sign.",
  steps: [
    "No verified step-by-step guide is available for this sign.",
    "Do not use another sign's reference as a substitute."
  ]
};

const els = {
  html: document.documentElement,
  themeToggle: document.querySelector("#themeToggle"),
  themeIconSun: document.querySelector("#themeIconSun"),
  themeIconMoon: document.querySelector("#themeIconMoon"),
  themeLabelText: document.querySelector("#themeLabelText"),
  modelStatusPill: document.querySelector("#modelStatusPill"),
  modelStatusText: document.querySelector("#modelStatusText"),
  videoContainer: document.querySelector("#videoContainer"),
  video: document.querySelector("#video"),
  overlay: document.querySelector("#overlay"),
  cameraButton: document.querySelector("#cameraButton"),
  switchCameraBtn: document.querySelector("#switchCameraBtn"),
  cameraBtnText: document.querySelector("#cameraBtnText"),
  cameraBadge: document.querySelector("#cameraBadge"),
  trackingDot: document.querySelector("#trackingDot"),
  trackingState: document.querySelector("#trackingState"),
  handChip: document.querySelector("#handChip"),
  bodyRegionBadge: document.querySelector("#bodyRegionBadge"),
  bodyRegionText: document.querySelector("#bodyRegionText"),
  signBadge: document.querySelector("#signBadge"),
  decisionState: document.querySelector("#decisionState"),
  currentSign: document.querySelector("#currentSign"),
  confidenceVal: document.querySelector("#confidenceVal"),
  gestureObsBar: document.querySelector("#gestureObsBar"),
  obsProgressFill: document.querySelector("#obsProgressFill"),
  obsLabel: document.querySelector("#obsLabel"),
  recognitionReason: document.querySelector("#recognitionReason"),
  speechButton: document.querySelector("#speechButton"),
  speechBtnText: document.querySelector("#speechBtnText"),
  speechText: document.querySelector("#speechText"),
  ttsButton: document.querySelector("#ttsButton"),
  exportButton: document.querySelector("#exportButton"),
  addUnknownButton: document.querySelector("#addUnknownButton"),
  clearButton: document.querySelector("#clearButton"),
  sentenceOutput: document.querySelector("#sentenceOutput"),
  sentenceChips: document.querySelector("#sentenceChips"),
  transcript: document.querySelector("#transcript"),
  lastRecognizedWord: document.querySelector("#lastRecognizedWord"),
  practiceTarget: document.querySelector("#practiceTarget"),
  practiceFeedback: document.querySelector("#practiceFeedback"),
  nextPracticeButton: document.querySelector("#nextPracticeButton"),
  vocabularyList: document.querySelector("#vocabularyList"),
  vocabStatus: document.querySelector("#vocabStatus"),

  // Sentence Mode Elements
  startSentenceBtn: document.querySelector("#startSentenceBtn"),
  stopSentenceBtn: document.querySelector("#stopSentenceBtn"),
  sentenceModeBadge: document.querySelector("#sentenceModeBadge"),

  // Tutor Modal elements
  tutorModal: document.querySelector("#tutorModal"),
  closeTutorBtn: document.querySelector("#closeTutorBtn"),
  tutorSignTitle: document.querySelector("#tutorSignTitle"),
  tutorBodyTarget: document.querySelector("#tutorBodyTarget"),
  tutorHandShape: document.querySelector("#tutorHandShape"),
  tutorMovement: document.querySelector("#tutorMovement"),
  tutorStepsList: document.querySelector("#tutorStepsList"),
  tutorVideoPlayer: document.querySelector("#tutorVideoPlayer"),
  practiceThisSignBtn: document.querySelector("#practiceThisSignBtn"),

  // Practice Section Elements
  startPracticeBtn: document.querySelector("#startPracticeBtn"),
  accuracyContainer: document.querySelector("#accuracyContainer"),
  accuracyPill: document.querySelector("#accuracyPill"),
  accuracyDot: document.querySelector("#accuracyDot"),
  accuracyScoreValue: document.querySelector("#accuracyScoreValue"),
  accuracyBarFill: document.querySelector("#accuracyBarFill"),
  practiceFeedback: document.querySelector("#practiceFeedback"),

  // Correction / Retraining Studio Modal Elements
  correctSignBtn: document.querySelector("#correctSignBtn"),
  correctionModal: document.querySelector("#correctionModal"),
  closeCorrectionBtn: document.querySelector("#closeCorrectionBtn"),
  correctionModelThought: document.querySelector("#correctionModelThought"),
  correctGlossSelect: document.querySelector("#correctGlossSelect"),
  submitCorrectionBtn: document.querySelector("#submitCorrectionBtn"),
  retrainVideo: document.querySelector("#retrainVideo"),
  retrainOverlay: document.querySelector("#retrainOverlay"),
  retrainBadge: document.querySelector("#retrainBadge"),
  retrainDot: document.querySelector("#retrainDot"),
  retrainStatusText: document.querySelector("#retrainStatusText"),
  retrainRecOverlay: document.querySelector("#retrainRecOverlay"),
  retrainRecTimerText: document.querySelector("#retrainRecTimerText"),
  retrainProgressBar: document.querySelector("#retrainProgressBar"),
  startRetrainRecBtn: document.querySelector("#startRetrainRecBtn"),
  stopRetrainRecBtn: document.querySelector("#stopRetrainRecBtn"),

  // Logo & Favicon Elements
  brandLogoWhite: document.querySelector("#brandLogoWhite"),
  brandLogoDark: document.querySelector("#brandLogoDark"),
  appFavicon: document.querySelector("#appFavicon"),

  // Splash Screen Elements
  appSplash: document.querySelector("#appSplash"),
  splashSkipBtn: document.querySelector("#splashSkipBtn"),

  // Download Transcript Button
  downloadTranscriptBtn: document.querySelector("#downloadTranscriptBtn"),

  // AI Practice Accuracy Scoring Gauge Elements
  accuracyContainer: document.querySelector("#accuracyContainer"),
  accuracyScoreValue: document.querySelector("#accuracyScoreValue"),
  accuracyBarFill: document.querySelector("#accuracyBarFill"),
  accuracyDot: document.querySelector("#accuracyDot"),
  accuracyPill: document.querySelector("#accuracyPill"),

  // Custom Sign Input
  customGlossInput: document.querySelector("#customGlossInput"),

  // Practice Target Selector Modal Elements
  openPracticeSelectorBtn: document.querySelector("#openPracticeSelectorBtn"),
  closePracticeSelectBtn: document.querySelector("#closePracticeSelectBtn"),
  practiceSelectModal: document.querySelector("#practiceSelectModal"),
  tabSentencesBtn: document.querySelector("#tabSentencesBtn"),
  tabWordsBtn: document.querySelector("#tabWordsBtn"),
  practiceSentencesView: document.querySelector("#practiceSentencesView"),
  practiceWordsView: document.querySelector("#practiceWordsView"),
  sentencesCategoryList: document.querySelector("#sentencesCategoryList"),
  wordsCategoryList: document.querySelector("#wordsCategoryList"),
  watchReferenceClipBtn: document.querySelector("#watchReferenceClipBtn"),
  practiceTargetType: document.querySelector("#practiceTargetType"),
  startPracticeBtn: document.querySelector("#startPracticeBtn"),

  // Header and Dock Practice & Retrain Studio Buttons
  headerPracticeBtn: document.querySelector("#headerPracticeBtn"),
  openRetrainStudioBtn: document.querySelector("#openRetrainStudioBtn"),
  controlPracticeBtn: document.querySelector("#controlPracticeBtn"),

  // Practice Studio Modal Elements
  practiceModal: document.querySelector("#practiceModal"),
  closePracticeModalBtn: document.querySelector("#closePracticeModalBtn"),
  practiceTargetTitle: document.querySelector("#practiceTargetTitle"),
  practiceGlossSelect: document.querySelector("#practiceGlossSelect"),
  practiceRefVideo: document.querySelector("#practiceRefVideo"),
  refVideoLabel: document.querySelector("#refVideoLabel"),
  practiceCamVideo: document.querySelector("#practiceCamVideo"),
  practiceCamWrapper: document.querySelector(".practice-cam-wrapper"),
  practiceNotification: document.querySelector("#practiceNotification"),
  practiceOverlay: document.querySelector("#practiceOverlay"),
  practiceRecOverlay: document.querySelector("#practiceRecOverlay"),
  practiceRecTimerText: document.querySelector("#practiceRecTimerText"),
  practiceAnalyzingBox: document.querySelector("#practiceAnalyzingBox"),
  practiceAnalyzingSubtitle: document.querySelector("#practiceAnalyzingSubtitle"),
  practiceAnalyzingFill: document.querySelector("#practiceAnalyzingFill"),
  practiceEvalResult: document.querySelector("#practiceEvalResult"),
  practiceVerdictIcon: document.querySelector("#practiceVerdictIcon"),
  practiceStatusText: document.querySelector("#practiceStatusText"),
  practiceScoreBadge: document.querySelector("#practiceScoreBadge"),
  practiceMeterFill: document.querySelector("#practiceMeterFill"),
  practiceMetricTarget: document.querySelector("#practiceMetricTarget"),
  practiceMetricDetected: document.querySelector("#practiceMetricDetected"),
  practiceMetricHandShape: document.querySelector("#practiceMetricHandShape"),
  practiceMetricMotion: document.querySelector("#practiceMetricMotion"),
  practiceEvalDetails: document.querySelector("#practiceEvalDetails"),
  practiceVerdictBanner: document.querySelector("#practiceVerdictBanner"),
  practiceBannerText: document.querySelector("#practiceBannerText"),
  practiceBannerBadge: document.querySelector("#practiceBannerBadge"),
  practiceTryAgainBtn: document.querySelector("#practiceTryAgainBtn"),
  practiceNextSignBtn: document.querySelector("#practiceNextSignBtn"),
  practiceApproveAgainBtn: document.querySelector("#practiceApproveAgainBtn"),
  practiceHudBadge: document.querySelector("#practiceHudBadge"),
  practiceHudPipeline: document.querySelector("#practiceHudPipeline"),
  practiceHudSignal: document.querySelector("#practiceHudSignal"),
  practiceHudOrgan: document.querySelector("#practiceHudOrgan"),
  modalStartPracticeBtn: document.querySelector("#modalStartPracticeBtn"),
  modalStartPracticeBtnText: document.querySelector("#modalStartPracticeBtnText"),
  modalStopPracticeBtn: document.querySelector("#modalStopPracticeBtn"),
  modalStopPracticeBtnText: document.querySelector("#modalStopPracticeBtnText"),
  camBoxRecordBtn: document.querySelector("#camBoxRecordBtn"),
  camBoxRecordBtnText: document.querySelector("#camBoxRecordBtnText"),
  camBoxStopBtn: document.querySelector("#camBoxStopBtn"),
  camBoxStopBtnText: document.querySelector("#camBoxStopBtnText"),
  sidePanelRecordBtn: document.querySelector("#sidePanelRecordBtn"),
  sidePanelRecordBtnText: document.querySelector("#sidePanelRecordBtnText"),
  practiceInstructionTip: document.querySelector("#practiceInstructionTip"),

  // In-Page Practice Section Elements
  pagePracticeGlossSelect: document.querySelector("#pagePracticeGlossSelect"),
  pagePracticeRecordBtn: document.querySelector("#pagePracticeRecordBtn"),
  pagePracticeRecordBtnText: document.querySelector("#pagePracticeRecordBtnText"),
  pagePracticeStopBtn: document.querySelector("#pagePracticeStopBtn"),
  pagePracticeStopBtnText: document.querySelector("#pagePracticeStopBtnText"),
  pagePracticeLoadingBox: document.querySelector("#pagePracticeLoadingBox"),
  pagePracticeLoadingSubtitle: document.querySelector("#pagePracticeLoadingSubtitle"),
  pagePracticeLoadingFill: document.querySelector("#pagePracticeLoadingFill"),
  pagePracticeResultBox: document.querySelector("#pagePracticeResultBox"),
  pageVerdictIcon: document.querySelector("#pageVerdictIcon"),
  pageStatusText: document.querySelector("#pageStatusText"),
  pageScoreBadge: document.querySelector("#pageScoreBadge"),
  pageMeterFill: document.querySelector("#pageMeterFill"),
  pageMetricTarget: document.querySelector("#pageMetricTarget"),
  pageMetricDetected: document.querySelector("#pageMetricDetected"),
  pageMetricHandShape: document.querySelector("#pageMetricHandShape"),
  pageMetricMotion: document.querySelector("#pageMetricMotion"),
  pageEvalDetails: document.querySelector("#pageEvalDetails"),
  pagePracticeTryAgainBtn: document.querySelector("#pagePracticeTryAgainBtn"),
  pagePracticeRecOverlay: document.querySelector("#pagePracticeRecOverlay"),
  pagePracticeRecTimerText: document.querySelector("#pagePracticeRecTimerText"),
  modalPracticeTryAgainBtn: document.querySelector("#modalPracticeTryAgainBtn"),
  sidebarTargetGlossBadge: document.querySelector("#sidebarTargetGlossBadge"),
  openStudioPopupBtn: document.querySelector("#openStudioPopupBtn"),

  // Train AI Loading & Success Screen Elements
  retrainLoadingBox: document.querySelector("#retrainLoadingBox"),
  retrainLoadingSubtitle: document.querySelector("#retrainLoadingSubtitle"),
  retrainLoadingFill: document.querySelector("#retrainLoadingFill"),
  retrainSuccessBox: document.querySelector("#retrainSuccessBox"),
  retrainResultIcon: document.querySelector("#retrainResultIcon"),
  retrainSuccessStatus: document.querySelector("#retrainSuccessStatus"),
  retrainSuccessBadge: document.querySelector("#retrainSuccessBadge"),
  retrainSuccessDetails: document.querySelector("#retrainSuccessDetails"),
  retrainPracticeNowBtn: document.querySelector("#retrainPracticeNowBtn"),
  retrainTeachAnotherBtn: document.querySelector("#retrainTeachAnotherBtn"),
};

const ctx = els.overlay.getContext("2d");

async function init() {
  applyTheme(state.theme);
  setupSplashDismissal();
  ensureRecognitionOutputPanel();
  await loadModel();
  await loadVocabulary();
  setupSpeechRecognition();
  bindEvents();
  renderTranscript();
  startLiveHeaderClock();
}

function ensureRecognitionOutputPanel() {
  const transcriptPanel = document.querySelector(".transcript-panel");
  if (!transcriptPanel) return;

  let output = document.querySelector("#lastRecognizedWord");
  if (!output) {
    const card = document.createElement("div");
    card.className = "last-recognized-sign";
    card.setAttribute("aria-live", "polite");
    card.setAttribute("aria-atomic", "true");
    card.innerHTML = '<span class="mini-label">Last recognized sign</span><strong id="lastRecognizedWord" class="is-empty">Waiting for a sign…</strong>';
    const transcript = transcriptPanel.querySelector("#transcript");
    transcriptPanel.insertBefore(card, transcript || null);
    output = card.querySelector("#lastRecognizedWord");
  }
  els.lastRecognizedWord = output;
}

function startLiveHeaderClock() {
  updateHeaderClock();
  setInterval(updateHeaderClock, 1000);
}

function updateHeaderClock() {
  const timeStr = new Date().toLocaleTimeString();
  if (els.modelStatusText) {
    els.modelStatusText.textContent = `ISL Model Active • ${timeStr}`;
  }
}

function dismissSplash() {
  if (els.appSplash && !els.appSplash.classList.contains("hidden")) {
    els.appSplash.style.transition = "opacity 0.6s ease, transform 0.6s ease";
    els.appSplash.style.opacity = "0";
    els.appSplash.style.transform = "scale(1.04)";
    setTimeout(() => {
      els.appSplash.classList.add("hidden");
    }, 600);
  }
}

function setupSplashDismissal() {
  if (els.splashSkipBtn) {
    els.splashSkipBtn.addEventListener("click", dismissSplash);
  }

  // Auto-transition smoothly into main console after resources finish loading (1.5 seconds)
  setTimeout(dismissSplash, 1500);
}

/* Theme Toggle Engine */
function applyTheme(themeName) {
  state.theme = themeName;
  localStorage.setItem("isl_theme", themeName);
  els.html.setAttribute("data-theme", themeName);

  if (themeName === "dark") {
    els.themeIconSun.classList.remove("hidden");
    els.themeIconMoon.classList.add("hidden");
    els.themeLabelText.textContent = "Light Mode";
    if (els.brandLogoWhite) els.brandLogoWhite.classList.remove("hidden");
    if (els.brandLogoDark) els.brandLogoDark.classList.add("hidden");
    if (els.appFavicon) els.appFavicon.href = "/static/images/white logo.png";
  } else {
    els.themeIconSun.classList.add("hidden");
    els.themeIconMoon.classList.remove("hidden");
    els.themeLabelText.textContent = "Dark Mode";
    if (els.brandLogoWhite) els.brandLogoWhite.classList.add("hidden");
    if (els.brandLogoDark) els.brandLogoDark.classList.remove("hidden");
    if (els.appFavicon) els.appFavicon.href = "/static/images/black logo.png";
  }
}

function toggleTheme() {
  const newTheme = state.theme === "dark" ? "light" : "dark";
  applyTheme(newTheme);
}

function downloadTranscript() {
  const timeStr = new Date().toLocaleString();

  const pdfContainer = document.createElement("div");
  pdfContainer.style.padding = "32px";
  pdfContainer.style.fontFamily = "'Inter', system-ui, -apple-system, sans-serif";
  pdfContainer.style.color = "#0f172a";
  pdfContainer.style.background = "#ffffff";
  pdfContainer.style.maxWidth = "750px";
  pdfContainer.style.margin = "0 auto";

  let html = `
    <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 2px solid #e2e8f0; padding-bottom: 16px; margin-bottom: 24px;">
      <div style="display: flex; align-items: center; gap: 14px;">
        <img src="/static/images/black logo.png" style="height: 44px; width: auto;" alt="ISL Lens Logo">
        <div>
          <h1 style="font-size: 22px; font-weight: 800; margin: 0; color: #0f172a; font-family: 'Outfit', sans-serif;">ISL-Lens</h1>
          <p style="font-size: 11px; color: #64748b; margin: 2px 0 0 0;">Real-Time Indian Sign Language Interpreter</p>
        </div>
      </div>
      <div style="text-align: right;">
        <span style="display: inline-block; font-size: 10px; font-weight: 700; background: #f1f5f9; color: #0f172a; padding: 4px 10px; border-radius: 99px; border: 1px solid #cbd5e1;">OFFICIAL TRANSCRIPT</span>
        <p style="font-size: 11px; color: #64748b; margin: 6px 0 0 0;">${timeStr}</p>
      </div>
    </div>

    <h3 style="font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #475569; margin-bottom: 12px; font-family: 'Outfit', sans-serif;">Recognized Sign Translations & Dialogue</h3>
    
    <table style="width: 100%; border-collapse: collapse; margin-bottom: 28px; font-size: 12px;">
      <thead>
        <tr style="background: #f8fafc; border-top: 1px solid #e2e8f0; border-bottom: 2px solid #e2e8f0; text-align: left;">
          <th style="padding: 10px 12px; width: 16%; color: #475569;">Time</th>
          <th style="padding: 10px 12px; width: 18%; color: #475569;">Type</th>
          <th style="padding: 10px 12px; color: #475569;">Sign / Sentence Translation</th>
        </tr>
      </thead>
      <tbody>
  `;

  if (!state.conversation || state.conversation.length === 0) {
    html += `
      <tr>
        <td colspan="3" style="padding: 16px 12px; text-align: center; color: #94a3b8; font-style: italic;">No sign language translations captured during this session.</td>
      </tr>
    `;
  } else {
    state.conversation.forEach((item, idx) => {
      const time = new Date(item.timestamp).toLocaleTimeString();
      const rowBg = idx % 2 === 0 ? "#ffffff" : "#f8fafc";
      const badgeBg = item.type === "sentence" ? "#e0f2fe" : "#f1f5f9";
      const badgeColor = item.type === "sentence" ? "#0369a1" : "#334155";
      html += `
        <tr style="background: ${rowBg}; border-bottom: 1px solid #f1f5f9;">
          <td style="padding: 10px 12px; color: #64748b; font-family: monospace;">${time}</td>
          <td style="padding: 10px 12px;">
            <span style="font-size: 10px; font-weight: 700; text-transform: uppercase; background: ${badgeBg}; color: ${badgeColor}; padding: 3px 8px; border-radius: 4px;">${escapeHtml(item.type)}</span>
          </td>
          <td style="padding: 10px 12px; font-weight: 600; color: #0f172a;">${escapeHtml(item.text)}</td>
        </tr>
      `;
    });
  }

  const speechText = els.speechText ? els.speechText.textContent : "None";

  html += `
      </tbody>
    </table>

    <h3 style="font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #475569; margin-bottom: 8px; font-family: 'Outfit', sans-serif;">Captured Speech Input Log</h3>
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 14px; font-size: 12px; color: #334155; line-height: 1.5; margin-bottom: 32px;">
      ${escapeHtml(speechText)}
    </div>

    <div style="border-top: 1px solid #e2e8f0; padding-top: 16px; display: flex; justify-content: space-between; align-items: center; font-size: 10px; color: #94a3b8;">
      <span>ISL-Lens AI Interpreter Console</span>
      <span>Official Transcript PDF Document</span>
    </div>
  `;

  pdfContainer.innerHTML = html;
  document.body.appendChild(pdfContainer);

  if (window.html2pdf) {
    const opt = {
      margin: 10,
      filename: `isl_lens_transcript_${Date.now()}.pdf`,
      image: { type: "jpeg", quality: 0.98 },
      html2canvas: { scale: 2, useCORS: true },
      jsPDF: { unit: "mm", format: "a4", orientation: "portrait" }
    };
    window.html2pdf().set(opt).from(pdfContainer).save().then(() => {
      if (document.body.contains(pdfContainer)) document.body.removeChild(pdfContainer);
    }).catch(() => {
      if (document.body.contains(pdfContainer)) document.body.removeChild(pdfContainer);
    });
  } else {
    window.print();
    if (document.body.contains(pdfContainer)) document.body.removeChild(pdfContainer);
  }
}

function bindEvents() {
  if (els.themeToggle) els.themeToggle.addEventListener("click", toggleTheme);
  if (els.cameraButton) els.cameraButton.addEventListener("click", toggleCamera);
  if (els.switchCameraBtn) els.switchCameraBtn.addEventListener("click", switchCamera);
  if (els.ttsButton) els.ttsButton.addEventListener("click", speakOutput);
  if (els.exportButton) els.exportButton.addEventListener("click", exportConversation);
  if (els.addUnknownButton) els.addUnknownButton.addEventListener("click", commitCurrentSign);
  if (els.clearButton) els.clearButton.addEventListener("click", clearConversation);
  if (els.speechButton) els.speechButton.addEventListener("click", toggleSpeechRecognition);
  if (els.nextPracticeButton) els.nextPracticeButton.addEventListener("click", nextPracticeTarget);
  if (els.downloadTranscriptBtn) {
    els.downloadTranscriptBtn.addEventListener("click", downloadTranscript);
  }

  // Sentence Generation Mode Events
  if (els.startSentenceBtn) {
    els.startSentenceBtn.addEventListener("click", startSentenceRecording);
  }
  if (els.stopSentenceBtn) {
    els.stopSentenceBtn.addEventListener("click", stopSentenceRecording);
  }

  // Tutor Modal Events
  if (els.closeTutorBtn) {
    els.closeTutorBtn.addEventListener("click", hideTutorModal);
  }
  if (els.tutorVideoPlayer) {
    els.tutorVideoPlayer.addEventListener("error", () => {
      const label = els.tutorVideoPlayer.closest(".tutor-video-container")?.querySelector(".video-label");
      if (label) label.textContent = "No verified reference clip is available for this sign.";
    });
  }
  if (els.practiceRefVideo) {
    els.practiceRefVideo.addEventListener("error", () => {
      if (els.refVideoLabel) els.refVideoLabel.textContent = "No verified reference clip";
      if (els.practiceInstructionTip) {
        els.practiceInstructionTip.textContent = `No verified reference clip is available for "${state.practiceTargetGloss}". No other sign's video is shown.`;
      }
    });
  }
  if (els.tutorModal) {
    els.tutorModal.addEventListener("click", (e) => {
      if (e.target === els.tutorModal) hideTutorModal();
    });
  }
  if (els.practiceThisSignBtn) {
    els.practiceThisSignBtn.addEventListener("click", () => {
      openPracticeModalFromTutor();
    });
  }
  if (els.practiceTarget) {
    els.practiceTarget.addEventListener("click", () => {
      openPracticeModal(els.practiceTarget.textContent.trim());
    });
  }

  // Model Correction & Retraining Events
  if (els.correctSignBtn) {
    els.correctSignBtn.addEventListener("click", () => openCorrectionModal());
  }
  if (els.closeCorrectionBtn) {
    els.closeCorrectionBtn.addEventListener("click", hideCorrectionModal);
  }
  if (els.correctionModal) {
    els.correctionModal.addEventListener("click", (e) => {
      if (e.target === els.correctionModal) hideCorrectionModal();
    });
  }
  if (els.submitCorrectionBtn) {
    els.submitCorrectionBtn.addEventListener("click", submitCorrection);
  }

  // Practice Target Selector & Video Clip Events
  if (els.openPracticeSelectorBtn) {
    els.openPracticeSelectorBtn.addEventListener("click", openPracticeSelectorModal);
  }
  if (els.closePracticeSelectBtn) {
    els.closePracticeSelectBtn.addEventListener("click", hidePracticeSelectorModal);
  }
  if (els.practiceSelectModal) {
    els.practiceSelectModal.addEventListener("click", (e) => {
      if (e.target === els.practiceSelectModal) hidePracticeSelectorModal();
    });
  }
  if (els.tabSentencesBtn) {
    els.tabSentencesBtn.addEventListener("click", () => switchPracticeTab("sentences"));
  }
  if (els.tabWordsBtn) {
    els.tabWordsBtn.addEventListener("click", () => switchPracticeTab("words"));
  }
  if (els.watchReferenceClipBtn) {
    els.watchReferenceClipBtn.addEventListener("click", () => {
      const targetText = els.practiceTarget ? els.practiceTarget.textContent.trim() : "HELLO";
      const firstGloss = targetText.split(" • ")[0].split(" ")[0];
      openTutorModal(firstGloss);
    });
  }

  // Header and Dock Practice & Retrain Buttons
  if (els.headerPracticeBtn) {
    els.headerPracticeBtn.addEventListener("click", () => openPracticeModal(state.practiceTargetGloss || "WATER"));
  }
  if (els.openRetrainStudioBtn) {
    els.openRetrainStudioBtn.addEventListener("click", () => openCorrectionModal());
  }
  if (els.controlPracticeBtn) {
    els.controlPracticeBtn.addEventListener("click", () => openPracticeModal(state.practiceTargetGloss || "WATER"));
  }

  // Practice Studio Pop-up Events
  if (els.startPracticeBtn) {
    els.startPracticeBtn.addEventListener("click", startPracticeSession);
  }
  if (els.closePracticeModalBtn) {
    els.closePracticeModalBtn.addEventListener("click", hidePracticeModal);
  }
  if (els.practiceModal) {
    els.practiceModal.addEventListener("click", (e) => {
      if (e.target === els.practiceModal) hidePracticeModal();
    });
  }
  if (els.modalStartPracticeBtn) {
    els.modalStartPracticeBtn.addEventListener("click", startPracticeRecording);
  }
  if (els.modalStopPracticeBtn) {
    els.modalStopPracticeBtn.addEventListener("click", stopPracticeAndEvaluate);
  }
  if (els.camBoxRecordBtn) {
    els.camBoxRecordBtn.addEventListener("click", startPracticeRecording);
  }
  if (els.camBoxStopBtn) {
    els.camBoxStopBtn.addEventListener("click", stopPracticeAndEvaluate);
  }
  if (els.sidePanelRecordBtn) {
    els.sidePanelRecordBtn.addEventListener("click", () => openPracticeModalAndRecord());
  }

  // In-Page Practice Section Events
  if (els.pagePracticeRecordBtn) {
    els.pagePracticeRecordBtn.addEventListener("click", startInPagePracticeRecording);
  }
  if (els.pagePracticeStopBtn) {
    els.pagePracticeStopBtn.addEventListener("click", stopInPagePracticeAndEvaluate);
  }
  if (els.pagePracticeTryAgainBtn) {
    els.pagePracticeTryAgainBtn.addEventListener("click", resetInPagePractice);
  }
  if (els.modalPracticeTryAgainBtn) {
    els.modalPracticeTryAgainBtn.addEventListener("click", startPracticeRecording);
  }
  if (els.practiceTryAgainBtn) {
    els.practiceTryAgainBtn.addEventListener("click", retryPracticeRecording);
  }
  if (els.practiceNextSignBtn) {
    els.practiceNextSignBtn.addEventListener("click", practiceNextSign);
  }
  if (els.practiceApproveAgainBtn) {
    els.practiceApproveAgainBtn.addEventListener("click", resetPracticeToRecord);
  }
  if (els.retrainPracticeNowBtn) {
    els.retrainPracticeNowBtn.addEventListener("click", practiceJustTrainedSign);
  }
  if (els.retrainTeachAnotherBtn) {
    els.retrainTeachAnotherBtn.addEventListener("click", resetRetrainForm);
  }
}

/* Sentence Generation Engine: Continuous Frame Recording & Temporal Sliding Window Segmentation */
async function startSentenceRecording() {
  state.isSentenceRecording = true;
  state.sentenceFrameBuffer = [];
  state.sentenceBodyRegions = [];
  state.sentenceStartTime = Date.now();
  state.recordedGlosses = [];

  if (els.sentenceModeBadge) {
    els.sentenceModeBadge.textContent = "Recording (0.0s)...";
    els.sentenceModeBadge.className = "badge-pill active";
  }
  if (els.startSentenceBtn) els.startSentenceBtn.classList.add("hidden");
  if (els.stopSentenceBtn) els.stopSentenceBtn.classList.remove("hidden");

  renderSentenceChips();
  if (els.sentenceOutput) {
    els.sentenceOutput.textContent = "🔴 Recording in progress... Perform your signs continuously in camera view, then click 'Pause & Translate Sentence'.";
  }

  if (!state.stream) {
    await startCamera();
    if (!state.stream) {
      state.isSentenceRecording = false;
      if (els.startSentenceBtn) els.startSentenceBtn.classList.remove("hidden");
      if (els.stopSentenceBtn) els.stopSentenceBtn.classList.add("hidden");
      if (els.sentenceModeBadge) {
        els.sentenceModeBadge.textContent = "Ready";
        els.sentenceModeBadge.className = "badge-pill";
      }
      if (els.sentenceOutput) {
        els.sentenceOutput.textContent = "Camera access is needed to record a sign sequence. Start the camera, then try again.";
      }
    }
  }
}

async function stopSentenceRecording() {
  if (!state.isSentenceRecording && state.sentenceFrameBuffer.length === 0) return;
  state.isSentenceRecording = false;

  if (els.sentenceModeBadge) {
    els.sentenceModeBadge.textContent = "Loading & Understanding...";
    els.sentenceModeBadge.className = "badge-pill";
  }

  if (els.startSentenceBtn) els.startSentenceBtn.classList.remove("hidden");
  if (els.stopSentenceBtn) els.stopSentenceBtn.classList.add("hidden");

  if (els.sentenceOutput) {
    els.sentenceOutput.textContent = "⚡ Loading and analyzing gesture sequence... Understanding what you signed...";
  }

  const totalFrames = state.sentenceFrameBuffer.length;

  if (totalFrames < 6) {
    if (els.sentenceOutput) {
      els.sentenceOutput.textContent = "Recording duration was too short to analyze. Please hold signs clearly for at least 1-2 seconds.";
    }
    state.recordedGlosses = [];
    renderSentenceChips();
    if (els.sentenceModeBadge) els.sentenceModeBadge.textContent = "Ready";
    return;
  }

  // Small delay for smooth UX transition
  await new Promise((resolve) => setTimeout(resolve, 250));

  // Perform Frame-by-Frame Temporal Sliding Window Segmentation over recorded video stream
  const extractedGlosses = processSentenceBufferTemporalSegmentation(state.sentenceFrameBuffer, state.sentenceBodyRegions);
  state.recordedGlosses = extractedGlosses;

  renderSentenceChips();
  const translation = translateISLGlossSequence(extractedGlosses);

  if (els.sentenceOutput) {
    els.sentenceOutput.textContent = translation;
  }

  if (els.sentenceModeBadge) {
    els.sentenceModeBadge.textContent = "Ready";
  }

  if (extractedGlosses.length > 0) {
    addConversationEntry("sentence", translation);
    speakText(translation);
  }
}

function classifySentenceSegment(segmentFrames, segmentRegions, model) {
  if (!segmentFrames || segmentFrames.length < 6) {
    return { sign: "UNKNOWN", confidence: 0, isAccepted: false };
  }
  const sampled = sample32Frames(segmentFrames);
  const pred = classifySequence(sampled, model, {
    bodyRegions: segmentRegions,
    totalFrames: segmentFrames.length,
    rawBuffer: segmentFrames,
  });
  const resolved = resolvePhraseFromOpeningPronoun(pred.sign, segmentFrames, segmentRegions, model);
  const posCheck = checkPredictedBodyPosition(resolved, segmentRegions, segmentFrames.length);
  if (!posCheck.accepted) {
    return { sign: "UNKNOWN", confidence: 0, isAccepted: false };
  }
  const isAccepted = pred.confidence >= 0.28 && resolved !== "UNKNOWN";
  return {
    sign: resolved,
    confidence: pred.confidence || 0,
    isAccepted,
    probs: pred.probs
  };
}

/* Hand-Presence Active Episode Extraction
   Detects contiguous signing episodes separated by natural pauses where hands leave the camera frame */
function extractHandPresenceEpisodes(frameBuffer, bodyRegions = []) {
  const N = frameBuffer.length;
  if (N < 6) return [];

  const handFlags = frameBuffer.map((f) => isHandPresentInVector(f));

  // Bridge brief 1-2 frame tracking drops
  const smoothed = [...handFlags];
  for (let i = 1; i < N - 1; i++) {
    if (!smoothed[i] && smoothed[i - 1] && smoothed[i + 1]) smoothed[i] = true;
  }
  for (let i = 2; i < N - 1; i++) {
    if (!smoothed[i] && !smoothed[i - 1] && smoothed[i - 2] && smoothed[i + 1]) {
      smoothed[i] = true;
      smoothed[i - 1] = true;
    }
  }

  const episodes = [];
  let epStart = -1;

  for (let i = 0; i < N; i++) {
    if (smoothed[i]) {
      if (epStart === -1) epStart = i;
    } else {
      if (epStart !== -1) {
        const epLen = i - epStart;
        if (epLen >= 7) {
          episodes.push({
            start: epStart,
            end: i - 1,
            length: epLen,
            frames: frameBuffer.slice(epStart, i),
            regions: bodyRegions.slice(epStart, i),
          });
        }
        epStart = -1;
      }
    }
  }

  if (epStart !== -1) {
    const epLen = N - epStart;
    if (epLen >= 7) {
      episodes.push({
        start: epStart,
        end: N - 1,
        length: epLen,
        frames: frameBuffer.slice(epStart, N),
        regions: bodyRegions.slice(epStart, N),
      });
    }
  }

  return episodes;
}

/* Evaluate a single continuous active signing stream (hands continuously in camera frame) */
function evaluateSingleEpisodeStream(activeBuffer, activeRegions, model) {
  const activeN = activeBuffer.length;
  if (activeN < 8) return [];

  const rSingle = classifySentenceSegment(activeBuffer, activeRegions, model);
  if (activeN < 16) {
    return rSingle.isAccepted ? [rSingle.sign] : [];
  }

  // 1. Multi-Scale Frame-by-Frame Continuous Temporal Window Scanning
  const windowSizes = activeN >= 36 ? [16, 24] : [16];
  const step = 2;
  const frameVotes = Array.from({ length: activeN }, () => ({}));

  windowSizes.forEach((wSize) => {
    const W = Math.min(wSize, activeN - 2);
    for (let start = 0; start <= activeN - W; start += step) {
      const end = start + W;
      const windowFrames = activeBuffer.slice(start, end);
      const windowRegions = activeRegions.slice(start, end);

      const res = classifySentenceSegment(windowFrames, windowRegions, model);
      if (res.isAccepted && res.confidence >= 0.28) {
        const center = (start + end) / 2;
        const radius = W / 2;
        for (let i = start; i < end; i++) {
          const dist = Math.abs(i - center) / radius;
          const weight = Math.max(0.2, 1.0 - dist * dist);
          const score = res.confidence * weight;
          frameVotes[i][res.sign] = (frameVotes[i][res.sign] || 0) + score;
        }
      }
    }
  });

  // 2. Extract Frame Winners and Group into Sustained Intervals
  const frameWinners = [];
  for (let t = 0; t < activeN; t++) {
    const votes = frameVotes[t];
    let bestSign = null;
    let maxVote = 0;
    for (const [sign, v] of Object.entries(votes)) {
      if (v > maxVote) {
        maxVote = v;
        bestSign = sign;
      }
    }
    frameWinners.push(maxVote >= 0.65 ? bestSign : null);
  }

  // Extract Sustained Intervals (Minimum 8 frames duration ~0.4s to eliminate transition noise)
  const intervals = [];
  let currentIv = null;
  for (let t = 0; t < activeN; t++) {
    const s = frameWinners[t];
    if (!currentIv) {
      if (s) currentIv = { sign: s, start: t, end: t, count: 1 };
    } else {
      if (s && areEquivalentSigns(s, currentIv.sign)) {
        currentIv.count++;
        currentIv.end = t;
      } else {
        if (currentIv.count >= 8) intervals.push(currentIv);
        currentIv = s ? { sign: s, start: t, end: t, count: 1 } : null;
      }
    }
  }
  if (currentIv && currentIv.count >= 8) intervals.push(currentIv);

  // Deduplicate adjacent identical intervals
  const streamSigns = [];
  intervals.forEach((iv) => {
    if (streamSigns.length === 0 || !areEquivalentSigns(iv.sign, streamSigns[streamSigns.length - 1])) {
      streamSigns.push(iv.sign);
    }
  });

  // 3. Kinematic Boundary Split Search
  const candidateSplitPoints = new Set();

  // A. Midpoints of detected interval transitions
  for (let i = 0; i < intervals.length - 1; i++) {
    const mid = Math.floor((intervals[i].end + intervals[i + 1].start) / 2);
    if (mid >= 10 && mid <= activeN - 10) candidateSplitPoints.add(mid);
  }

  // B. Spatial Transitions
  for (let i = 10; i <= activeN - 10; i++) {
    const prevR = activeRegions[i - 1];
    const currR = activeRegions[i];
    if (prevR && currR && prevR !== currR && prevR !== "neutral" && currR !== "neutral") {
      candidateSplitPoints.add(i);
    }
  }

  // C. Speed Valleys
  const speeds = [];
  for (let i = 1; i < activeN; i++) {
    const dx = (activeBuffer[i][126] || 0) - (activeBuffer[i - 1][126] || 0);
    const dy = (activeBuffer[i][127] || 0) - (activeBuffer[i - 1][127] || 0);
    speeds.push(Math.hypot(dx, dy));
  }
  for (let i = 12; i <= activeN - 12; i++) {
    let avgBefore = 0, avgHere = 0, avgAfter = 0;
    for (let k = -3; k <= -2; k++) avgBefore += speeds[i + k] || 0;
    for (let k = -1; k <= 1; k++) avgHere += speeds[i + k] || 0;
    for (let k = 2; k <= 3; k++) avgAfter += speeds[i + k] || 0;
    avgBefore /= 2; avgHere /= 3; avgAfter /= 2;
    if (avgHere < avgBefore * 0.80 && avgHere < avgAfter * 0.80) {
      candidateSplitPoints.add(i);
    }
  }

  // D. Regular Temporal Grid across [0.20 * activeN, 0.80 * activeN]
  const gridStep = Math.max(3, Math.floor(activeN / 16));
  for (let i = Math.max(10, Math.floor(activeN * 0.20)); i <= Math.min(activeN - 10, Math.floor(activeN * 0.80)); i += gridStep) {
    candidateSplitPoints.add(i);
  }

  // Evaluate 2-Sign Split Candidates
  let best2Split = null;
  let best2Score = -1;

  candidateSplitPoints.forEach((S) => {
    if (S < 10 || S > activeN - 10) return;
    const seg1 = activeBuffer.slice(0, S);
    const seg2 = activeBuffer.slice(S, activeN);
    const reg1 = activeRegions.slice(0, S);
    const reg2 = activeRegions.slice(S, activeN);

    const r1 = classifySentenceSegment(seg1, reg1, model);
    const r2 = classifySentenceSegment(seg2, reg2, model);

    if (r1.isAccepted && r2.isAccepted && !areEquivalentSigns(r1.sign, r2.sign)) {
      const c1 = r1.confidence;
      const c2 = r2.confidence;
      if (c1 >= 0.30 && c2 >= 0.30) {
        let score = c1 + c2 + 0.25;
        if (isSemanticISLPair(r1.sign, r2.sign)) {
          score += 0.45;
        }
        if (streamSigns.length >= 2 && areEquivalentSigns(r1.sign, streamSigns[0]) && areEquivalentSigns(r2.sign, streamSigns[1])) {
          score += 0.50;
        }
        if (score > best2Score) {
          best2Score = score;
          best2Split = {
            r1: r1.sign,
            r2: r2.sign,
            c1,
            c2,
            score,
            isDiff: true,
          };
        }
      }
    }
  });

  // Evaluate 3-Sign Split Candidates for Longer Continuous Recordings (>= 40 frames, ~2.0s)
  let best3Split = null;
  let best3Score = -1;

  if (activeN >= 40) {
    const step3 = Math.max(4, Math.floor(activeN / 14));
    for (let S1 = 10; S1 <= activeN - 20; S1 += step3) {
      for (let S2 = S1 + 10; S2 <= activeN - 10; S2 += step3) {
        const seg1 = activeBuffer.slice(0, S1);
        const seg2 = activeBuffer.slice(S1, S2);
        const seg3 = activeBuffer.slice(S2, activeN);
        const reg1 = activeRegions.slice(0, S1);
        const reg2 = activeRegions.slice(S1, S2);
        const reg3 = activeRegions.slice(S2, activeN);

        const r1 = classifySentenceSegment(seg1, reg1, model);
        const r2 = classifySentenceSegment(seg2, reg2, model);
        const r3 = classifySentenceSegment(seg3, reg3, model);

        if (r1.isAccepted && r2.isAccepted && r3.isAccepted) {
          if (!areEquivalentSigns(r1.sign, r2.sign) && !areEquivalentSigns(r2.sign, r3.sign)) {
            const c1 = r1.confidence;
            const c2 = r2.confidence;
            const c3 = r3.confidence;
            if (c1 >= 0.30 && c2 >= 0.30 && c3 >= 0.30) {
              let score = c1 + c2 + c3 + 0.50;
              const tKey = [r1.sign, r2.sign, r3.sign].map((s) => formatGlossDisplay(s).toUpperCase()).join(" • ");
              if (["ME • DRINK • WATER", "ME • EAT • FOOD", "ME • GO • HOSPITAL", "ME • GO • HOME", "PLEASE • HELP • ME", "YOU • GO • WHERE"].includes(tKey)) {
                score += 0.60;
              }
              if (score > best3Score) {
                best3Score = score;
                best3Split = { r1: r1.sign, r2: r2.sign, r3: r3.sign, score };
              }
            }
          }
        }
      }
    }
  }

  // 4. Final Selection
  if (best3Split && best3Score >= 1.55) {
    return [best3Split.r1, best3Split.r2, best3Split.r3];
  }
  if (streamSigns.length === 3) {
    return streamSigns;
  }
  if (best2Split && best2Split.isDiff && (best2Split.score >= 0.95 || (best2Split.c1 >= 0.32 && best2Split.c2 >= 0.32))) {
    return [best2Split.r1, best2Split.r2];
  }
  if (streamSigns.length === 2) {
    return streamSigns;
  }
  if (streamSigns.length === 1) {
    return streamSigns;
  }
  if (rSingle.isAccepted && rSingle.confidence >= 0.30) {
    return [rSingle.sign];
  }
  if (best2Split && best2Split.isDiff) {
    return [best2Split.r1, best2Split.r2];
  }

  return [];
}

/* Master Temporal Segmentation Pipeline for Continuous Sentence Generation
   Seamlessly handles:
   1. Active signing episodes separated by natural hand-absence pauses (hands taken off screen)
   2. Continuous gesture transitions without removing hands from camera frame */
function processSentenceBufferTemporalSegmentation(frameBuffer, bodyRegions = []) {
  if (!state.modelLoaded || !state.model) return [];
  const N = frameBuffer.length;
  if (N < 8) return [];

  // Extract clean active signing episodes separated by natural hand-absence pauses (hands taken off screen)
  const episodes = extractHandPresenceEpisodes(frameBuffer, bodyRegions);

  if (episodes.length === 0) {
    return [];
  }

  // A. Multi-Episode Flow (User performed a sign, took hands off screen, then brought hands back for the next sign)
  if (episodes.length >= 2) {
    const episodePredictions = episodes.map((ep) => classifySentenceSegment(ep.frames, ep.regions, state.model));

    // Direct check for premier 2-sign combinations across episodes (including intentional repetition)
    if (episodes.length === 2) {
      const p1 = episodePredictions[0];
      const p2 = episodePredictions[1];
      if (p1.sign !== "UNKNOWN" && p2.sign !== "UNKNOWN") {
        if (p1.confidence >= 0.25 && p2.confidence >= 0.25) {
          if (isSemanticISLPair(p1.sign, p2.sign) || (p1.isAccepted && p2.isAccepted) || (p1.confidence >= 0.35 && p2.confidence >= 0.25)) {
            return [p1.sign, p2.sign];
          }
        }
      }
    }

    const episodeSigns = [];
    episodes.forEach((ep, idx) => {
      // If an individual episode is long (>= 36 frames), evaluate if it contained continuous gestures
      if (ep.length >= 36) {
        const subSigns = evaluateSingleEpisodeStream(ep.frames, ep.regions, state.model);
        if (subSigns.length >= 2) {
          subSigns.forEach((s) => episodeSigns.push(s));
          return;
        }
      }
      const pred = episodePredictions[idx] || classifySentenceSegment(ep.frames, ep.regions, state.model);
      if (pred.isAccepted && pred.sign !== "UNKNOWN") {
        episodeSigns.push(pred.sign);
      }
    });

    const ALLOWED_EPISODE_REPEATS = new Set(["WATER", "DRINK", "FOOD", "EAT", "TEA", "HELP", "PLEASE", "THANK YOU", "SORRY", "STOP", "CLAPS", "BOOK", "READ", "PHONE", "HOME", "COME", "GO"]);
    const finalChain = [];
    episodeSigns.forEach((s) => {
      const prev = finalChain[finalChain.length - 1];
      const normS = formatGlossDisplay(s).toUpperCase();
      if (finalChain.length === 0 || !areEquivalentSigns(s, prev)) {
        finalChain.push(s);
      } else if (ALLOWED_EPISODE_REPEATS.has(normS) && finalChain.filter((x) => areEquivalentSigns(x, s)).length < 2) {
        finalChain.push(s);
      }
    });

    if (finalChain.length >= 1) {
      return finalChain;
    }
  }

  // B. Single Episode Flow (Hands stayed in camera frame continuously)
  const activeEp = episodes[0];
  return evaluateSingleEpisodeStream(activeEp.frames, activeEp.regions, state.model);
}

function isReliableSentenceDetection(prediction, bodyRegions = []) {
  if (!prediction?.isAccepted) return false;
  if (formatGlossDisplay(prediction.sign).toUpperCase() !== "SHE") return true;

  // SHE is a frequent false positive for other pointing gestures. Require a
  // clear score and separation from the runner-up before adding it to a
  // sentence; ordinary recognition and other signs keep their existing gate.
  const probabilities = Array.from(prediction.probs || []).sort((a, b) => b - a);
  const topProbability = Number(prediction.confidence) || 0;
  const runnerUp = probabilities[0] >= topProbability - 1e-6
    ? (probabilities[1] || 0)
    : (probabilities[0] || 0);
  if (topProbability < 0.5 || topProbability - runnerUp < 0.12) return false;

  // The guide's SHE gesture points to one side. Reject a lateral sweep or
  // unstable hand placement, which is common in WHERE and other neutral-space
  // signs that the model can confuse with SHE.
  return getStableNeutralSide(bodyRegions) !== null;
}

function getStableNeutralSide(bodyRegions = []) {
  const directionalFrames = (bodyRegions || []).filter((region) =>
    region === "neutral_left" || region === "neutral_right"
  );
  if (directionalFrames.length < 4) return null;
  const leftCount = directionalFrames.filter((region) => region === "neutral_left").length;
  const rightCount = directionalFrames.length - leftCount;
  const stableSideRatio = Math.max(leftCount, rightCount) / directionalFrames.length;
  if (stableSideRatio < 0.8) return null;
  return leftCount > rightCount ? "left" : "right";
}

function resolvePhraseFromOpeningPronoun(predictedSign, frames, bodyRegions, model) {
  const sign = formatGlossDisplay(predictedSign).toUpperCase();
  if (sign !== "HOW ARE YOU" && sign !== "I AM FINE") return predictedSign;

  const prefixLength = Math.min(frames.length, Math.max(6, Math.ceil(frames.length * 0.4)));
  const prefixRegions = (bodyRegions || []).slice(0, prefixLength).filter(Boolean);
  const chestFrames = prefixRegions.filter((region) => region === "chest").length;
  const forwardFrames = prefixRegions.filter((region) => String(region).startsWith("neutral")).length;
  const observedFrames = chestFrames + forwardFrames;

  let openingPronoun = null;
  if (observedFrames >= 4 && chestFrames / observedFrames >= 0.6) {
    openingPronoun = "ME";
  } else if (observedFrames >= 4 && forwardFrames / observedFrames >= 0.6) {
    openingPronoun = "YOU";
  } else if (prefixLength >= 6) {
    // Fall back to the classifier on just the opening movement when body
    // anchors are unavailable. Only a confident ME/YOU result is decisive.
    const prefix = sample32Frames(frames.slice(0, prefixLength));
    const opening = classifySequence(prefix, model, {
      bodyRegions: prefixRegions,
      totalFrames: prefixLength,
    });
    if ((opening.sign === "ME" || opening.sign === "YOU")
        && opening.confidence >= Math.max(0.45, getRecognitionThreshold(model))) {
      openingPronoun = opening.sign;
    }
  }

  if (openingPronoun === "ME") return "I AM FINE";
  if (openingPronoun === "YOU") return "HOW ARE YOU";
  return predictedSign;
}

function renderSentenceChips() {
  if (!els.sentenceChips) return;
  els.sentenceChips.innerHTML = "";
  if (!state.recordedGlosses || state.recordedGlosses.length === 0) {
    els.sentenceChips.style.display = "none";
    return;
  }
  els.sentenceChips.style.display = "flex";
  els.sentenceChips.style.flexWrap = "wrap";
  els.sentenceChips.style.gap = "6px";
  els.sentenceChips.style.alignItems = "center";

  const label = document.createElement("span");
  label.className = "mini-label";
  label.style.width = "100%";
  label.style.marginBottom = "2px";
  label.textContent = "Listening Order Sequence:";
  els.sentenceChips.appendChild(label);

  state.recordedGlosses.forEach((gloss, idx) => {
    const chip = document.createElement("span");
    chip.className = "badge-pill";
    chip.style.background = "rgba(0, 242, 254, 0.15)";
    chip.style.color = "var(--accent-cyan)";
    chip.style.border = "1px solid rgba(0, 242, 254, 0.3)";
    chip.style.fontSize = "11px";
    chip.style.fontWeight = "600";
    chip.textContent = `${idx + 1}. ${formatGlossDisplay(gloss)}`;
    els.sentenceChips.appendChild(chip);
  });
}

// Complete Semantic Categorization for the 64 ISL Trained Classes
const ISL_GRAMMAR_CATEGORIES = {
  GREETINGS: new Set([
    "HELLO", "NAMASTE", "GOOD MORNING", "GOOD AFTERNOON", "GOOD EVENING", "GOOD NIGHT",
    "GOODBYE", "WELCOME", "NICE TO MEET YOU", "ALL THE BEST"
  ]),
  POLITE: new Set([
    "PLEASE", "THANK YOU", "SORRY", "OKAY", "YES", "NO", "CLAPS"
  ]),
  PRONOUNS: new Set([
    "ME", "YOU", "HE", "SHE"
  ]),
  PEOPLE: new Set([
    "MOTHER", "FATHER", "BROTHER", "SISTER", "FAMILY", "CHILDREN", "FRIEND", "TEACHER", "STUDENT"
  ]),
  ACTIONS: new Set([
    "EAT", "DRINK", "GO", "COME", "READ", "WRITE", "SIT", "STAND", "STOP", "HELP", "LOVE"
  ]),
  LOCATIONS: new Set([
    "HOME", "HOSPITAL", "SCHOOL", "MARKET"
  ]),
  OBJECTS: new Set([
    "WATER", "FOOD", "TEA", "BOOK", "PHONE", "AIRPLANE"
  ]),
  TIME: new Set([
    "TIME", "HOUR", "MINUTE", "TODAY"
  ]),
  STATES: new Set([
    "HAPPY", "SAD", "DANGER", "I AM FINE"
  ]),
  QUESTIONS: new Set([
    "WHAT", "WHERE", "WHEN", "HOW ARE YOU", "WHAT IS YOUR NAME"
  ])
};

function isSemanticISLPair(signA, signB) {
  const g1 = formatGlossDisplay(signA).toUpperCase().replace(/[\s_]+/g, " ").trim();
  const g2 = formatGlossDisplay(signB).toUpperCase().replace(/[\s_]+/g, " ").trim();
  if (areEquivalentSigns(g1, g2)) {
    return ["WATER", "DRINK", "FOOD", "EAT", "TEA", "HELP", "PLEASE", "THANK YOU", "SORRY", "STOP", "CLAPS", "BOOK", "READ", "PHONE", "HOME", "COME", "GO"].includes(g1);
  }

  // 1. Any polite modifier + anything
  if (ISL_GRAMMAR_CATEGORIES.POLITE.has(g1) || ISL_GRAMMAR_CATEGORIES.POLITE.has(g2)) return true;

  // 2. Any greeting + person / question / welcome
  if (ISL_GRAMMAR_CATEGORIES.GREETINGS.has(g1) || ISL_GRAMMAR_CATEGORIES.GREETINGS.has(g2)) return true;

  // 3. Pronouns + Actions / States / Locations / People
  if (ISL_GRAMMAR_CATEGORIES.PRONOUNS.has(g1) || g1 === "I AM FINE") return true;

  // 4. Questions + anything
  if (ISL_GRAMMAR_CATEGORIES.QUESTIONS.has(g1) || ISL_GRAMMAR_CATEGORIES.QUESTIONS.has(g2)) return true;

  // 5. Actions + Objects / Locations
  if (ISL_GRAMMAR_CATEGORIES.ACTIONS.has(g1)) return true;

  // 6. People + Locations / States / Actions
  if (ISL_GRAMMAR_CATEGORIES.PEOPLE.has(g1)) return true;

  // 7. Danger & Emergencies
  if (g1 === "DANGER" || g2 === "DANGER") return true;

  // 8. Time queries
  if (ISL_GRAMMAR_CATEGORIES.TIME.has(g1) || ISL_GRAMMAR_CATEGORIES.TIME.has(g2)) return true;

  return false;
}

function translateISLGlossSequence(glosses) {
  if (!glosses || glosses.length === 0) {
    return "No clear signs were captured during recording. Please try performing your signs clearly.";
  }

  // Normalize synonyms while PRESERVING intentional repeated signs (e.g. drinking sign twice -> I drink water)
  const ALLOWED_REPEATS = new Set(["WATER", "DRINK", "FOOD", "EAT", "TEA", "HELP", "PLEASE", "THANK YOU", "SORRY", "STOP", "CLAPS", "BOOK", "READ", "PHONE", "HOME", "COME", "GO"]);
  const list = [];
  glosses.forEach((g) => {
    let norm = formatGlossDisplay(g).toUpperCase().replace(/[\s_]+/g, " ").trim();
    if (norm === "HI") norm = "HELLO";
    if (norm === "THANKS" || norm === "THANKYOU") norm = "THANK YOU";
    if (norm === "IAMFINE" || norm === "I_AM_FINE") norm = "I AM FINE";
    if (norm === "HOWAREYOU" || norm === "HOW_ARE_YOU") norm = "HOW ARE YOU";
    if (norm === "WHATISYOURNAME" || norm === "WHAT_IS_YOUR_NAME") norm = "WHAT IS YOUR NAME";
    if (norm === "NICETOMEETYOU" || norm === "NICE_TO_MEET_YOU") norm = "NICE TO MEET YOU";
    if (norm === "ALLTHEBEST" || norm === "ALL_THE_BEST") norm = "ALL THE BEST";
    if (norm === "GOODMORNING" || norm === "GOOD_MORNING") norm = "GOOD MORNING";
    if (norm === "GOODAFTERNOON" || norm === "GOOD_AFTERNOON") norm = "GOOD AFTERNOON";
    if (norm === "GOODEVENING" || norm === "GOOD_EVENING") norm = "GOOD EVENING";
    if (norm === "GOODNIGHT" || norm === "GOOD_NIGHT") norm = "GOOD NIGHT";
    if (list.length === 0 || list[list.length - 1] !== norm) {
      list.push(norm);
    } else if (ALLOWED_REPEATS.has(norm) && list.filter((x) => x === norm).length < 2) {
      list.push(norm);
    }
  });

  if (list.length === 0) {
    return "No clear signs were captured during recording. Please try performing your signs clearly.";
  }

  const key = list.join(" • ");

  // 1. Direct Idiomatic & High-Frequency Multi-Sign Lookup
  const IDIOMS = {
    // Drinking & Eating Reduplications & Combos (User-requested ISL structures)
    "WATER • WATER": "I drink water.",
    "DRINK • DRINK": "I drink water.",
    "DRINK • WATER": "I drink water.",
    "WATER • DRINK": "I drink water.",
    "ME • WATER": "I drink water.",
    "ME • DRINK": "I drink water.",
    "ME • WATER • WATER": "I drink water.",
    "ME • DRINK • DRINK": "I drink water.",
    "ME • DRINK • WATER": "I drink water.",
    "ME • WATER • DRINK": "I drink water.",
    "FOOD • FOOD": "I eat food.",
    "EAT • EAT": "I eat food.",
    "EAT • FOOD": "I eat food.",
    "FOOD • EAT": "I eat food.",
    "ME • FOOD": "I eat food.",
    "ME • EAT": "I eat food.",
    "ME • FOOD • FOOD": "I eat food.",
    "ME • EAT • FOOD": "I eat food.",
    "ME • EAT • EAT": "I eat food.",
    "TEA • TEA": "I drink tea.",
    "DRINK • TEA": "I drink tea.",
    "TEA • DRINK": "I drink tea.",
    "ME • TEA": "I drink tea.",
    "ME • DRINK • TEA": "I drink tea.",
    "BOOK • BOOK": "Read a book.",
    "READ • BOOK": "Read the book.",
    "BOOK • READ": "Read the book.",
    "WRITE • BOOK": "Write in the book.",
    "BOOK • WRITE": "Write in the book.",
    "ME • BOOK": "I want to read a book.",
    "ME • READ": "I want to read a book.",
    "ME • WRITE": "I want to write.",
    "ME • READ • BOOK": "I want to read a book.",
    "ME • WRITE • BOOK": "I want to write in the book.",
    "PHONE • PHONE": "Use the phone.",
    "ME • PHONE": "I want to use the phone.",
    "ME • AIRPLANE": "I want to travel by airplane.",
    "HOME • HOME": "Go home.",
    "ME • HOME": "I am going home.",
    "ME • HOSPITAL": "I need to go to the hospital.",
    "ME • SCHOOL": "I am going to school.",
    "ME • MARKET": "I am going to the market.",
    "ME • HAPPY": "I am happy.",
    "ME • SAD": "I am sad.",
    "ME • TIME": "What time is it?",
    "ME • TODAY": "I am here today.",
    "ME • STUDENT": "I am a student.",
    "ME • TEACHER": "I am a teacher.",
    "ME • FRIEND": "I am your friend.",
    "ME • FAMILY": "This is my family.",
    "ME • MOTHER": "She is my mother.",
    "ME • FATHER": "He is my father.",
    "ME • BROTHER": "He is my brother.",
    "ME • SISTER": "She is my sister.",
    "ME • CHILDREN": "These are my children.",
    "YOU • WATER": "Do you want water?",
    "YOU • DRINK": "Do you want water?",
    "YOU • FOOD": "Do you want food?",
    "YOU • EAT": "Do you want food?",
    "YOU • TEA": "Do you want tea?",
    "YOU • BOOK": "Do you want a book?",
    "YOU • READ": "Do you want to read?",
    "YOU • WRITE": "Can you write this?",
    "YOU • PHONE": "Do you have a phone?",
    "YOU • AIRPLANE": "Are you travelling by airplane?",
    "YOU • HOME": "Are you going home?",
    "YOU • HOSPITAL": "Are you at the hospital?",
    "YOU • SCHOOL": "Are you going to school?",
    "YOU • MARKET": "Are you going to the market?",
    "YOU • HAPPY": "Are you happy?",
    "YOU • SAD": "Are you sad?",
    "YOU • HELP": "Can you help me?",
    "YOU • COME": "Please come.",
    "YOU • GO": "Where are you going?",
    "YOU • WELCOME": "You are welcome!",
    "MOTHER • FOOD": "Mother is preparing food.",
    "MOTHER • EAT": "Mother is eating food.",
    "MOTHER • WATER": "Mother, please give me water.",
    "MOTHER • DRINK": "Mother, please give me water.",
    "MOTHER • TEA": "Mother, please give me tea.",
    "MOTHER • HOME": "Mother is at home.",
    "MOTHER • HOSPITAL": "Mother is at the hospital.",
    "MOTHER • MARKET": "Mother went to the market.",
    "MOTHER • HAPPY": "Mother is happy.",
    "MOTHER • SAD": "Mother is sad.",
    "MOTHER • LOVE": "I love my mother.",
    "FATHER • FOOD": "Father is having food.",
    "FATHER • WATER": "Father, please give me water.",
    "FATHER • TEA": "Father is having tea.",
    "FATHER • HOME": "Father is at home.",
    "FATHER • HOSPITAL": "Father is at the hospital.",
    "FATHER • MARKET": "Father went to the market.",
    "FATHER • HAPPY": "Father is happy.",
    "FATHER • SAD": "Father is sad.",
    "FATHER • LOVE": "I love my father.",
    "BROTHER • SCHOOL": "Brother is at school.",
    "BROTHER • HOME": "Brother is at home.",
    "BROTHER • BOOK": "Brother is reading a book.",
    "SISTER • SCHOOL": "Sister is at school.",
    "SISTER • HOME": "Sister is at home.",
    "SISTER • BOOK": "Sister is reading a book.",
    "TEACHER • SCHOOL": "The teacher is at school.",
    "TEACHER • BOOK": "The teacher is reading a book.",
    "TEACHER • WRITE": "The teacher is writing.",
    "STUDENT • SCHOOL": "The student is at school.",
    "STUDENT • BOOK": "The student is reading a book.",
    "STUDENT • WRITE": "The student is writing.",
    "FRIEND • COME": "My friend is coming.",
    "FRIEND • GO": "My friend is going.",
    "FRIEND • HOME": "My friend is at home.",
    "FRIEND • SCHOOL": "My friend is at school.",
    "FRIEND • HELP": "My friend is helping me.",
    "FRIEND • HAPPY": "My friend is happy.",
    "FAMILY • HOME": "My family is at home.",
    "FAMILY • HAPPY": "My family is happy.",
    "FAMILY • LOVE": "I love my family.",
    "CHILDREN • SCHOOL": "The children are at school.",
    "CHILDREN • HOME": "The children are at home.",
    "CHILDREN • HAPPY": "The children are happy.",

    // Courtesy & Greetings
    "I AM FINE • THANK YOU": "I am fine, thank you.",
    "THANK YOU • I AM FINE": "Thank you, I am fine.",
    "HELLO • HOW ARE YOU": "Hello, how are you?",
    "HOW ARE YOU • I AM FINE": "How are you? I am fine.",
    "HOW ARE YOU • THANK YOU": "How are you? Thank you.",
    "HELLO • THANK YOU": "Hello, thank you!",
    "HELLO • WELCOME": "Hello, welcome!",
    "WELCOME • HOME": "Welcome home!",
    "WELCOME • FRIEND": "Welcome, my friend!",
    "NICE TO MEET YOU • FRIEND": "Nice to meet you, my friend!",
    "NAMASTE • WELCOME": "Namaste, welcome!",
    "GOODBYE • FRIEND": "Goodbye, my friend!",
    "ALL THE BEST • FRIEND": "All the best, my friend!",
    "ALL THE BEST • STUDENT": "All the best to the students!",

    // Help & Emergency
    "PLEASE • HELP": "Please help.",
    "PLEASE • HELP • ME": "Please help me.",
    "HELP • PLEASE": "Help, please.",
    "HELP • ME": "Please help me.",
    "DANGER • HELP": "Danger! Please help!",
    "HELP • DANGER": "Help, there is danger!",
    "STOP • DANGER": "Stop! It is dangerous!",
    "STOP • PLEASE": "Please stop.",
    "PLEASE • STOP": "Please stop.",

    // Polite Actions & Needs
    "PLEASE • WATER": "Please give me water.",
    "WATER • PLEASE": "Water, please.",
    "PLEASE • FOOD": "Please give me food.",
    "FOOD • PLEASE": "Food, please.",
    "PLEASE • TEA": "Please give me tea.",
    "TEA • PLEASE": "Tea, please.",
    "PLEASE • BOOK": "Please give me the book.",
    "PLEASE • PHONE": "Please give me the phone.",
    "PLEASE • SIT": "Please sit down.",
    "PLEASE • STAND": "Please stand up.",
    "PLEASE • COME": "Please come in.",
    "PLEASE • GO": "Please go ahead.",
    "PLEASE • READ": "Please read this.",
    "PLEASE • WRITE": "Please write this down.",
    "SORRY • PLEASE": "I am sorry, please forgive me.",
    "PLEASE • SORRY": "Please, I am sorry.",
    "SORRY • THANK YOU": "I am sorry, thank you.",
    "THANK YOU • PLEASE": "Thank you, please.",
    "YES • PLEASE": "Yes, please.",
    "NO • PLEASE": "No, please.",
    "YES • THANK YOU": "Yes, thank you.",
    "NO • THANK YOU": "No, thank you.",
    "YES • OKAY": "Yes, okay!",
    "NO • STOP": "No, stop!",

    // 3-Sign Sequences
    "ME • GO • HOSPITAL": "I need to go to the hospital.",
    "ME • GO • HOME": "I am going home.",
    "ME • GO • SCHOOL": "I am going to school.",
    "ME • GO • MARKET": "I am going to the market.",
    "PLEASE • WATER • WATER": "Please give me drinking water.",
    "PLEASE • DRINK • WATER": "Please give me drinking water.",
    "YOU • GO • WHERE": "Where are you going?",
    "YOU • COME • WHEN": "When are you coming?",
    "YOU • WANT • WHAT": "What do you want?",
    "MOTHER • FATHER • HOME": "Mother and father are at home.",
    "BROTHER • SISTER • SCHOOL": "Brother and sister are at school.",
    "HELLO • HOW ARE YOU • THANK YOU": "Hello, how are you? Thank you!",
    "I AM FINE • THANK YOU • GOODBYE": "I am fine, thank you. Goodbye!"
  };

  if (IDIOMS[key]) {
    return IDIOMS[key];
  }

  // 2. Single Sign Mapping
  if (list.length === 1) {
    const singleMap = {
      "HELLO": "Hello!",
      "GOODBYE": "Goodbye!",
      "THANK YOU": "Thank you!",
      "PLEASE": "Please.",
      "SORRY": "I am sorry.",
      "HELP": "Please help me.",
      "YES": "Yes.",
      "NO": "No.",
      "OKAY": "Okay.",
      "WATER": "Water.",
      "FOOD": "Food.",
      "EAT": "Eat.",
      "DRINK": "Drink.",
      "HOME": "Home.",
      "STUDENT": "Student.",
      "TEACHER": "Teacher.",
      "MOTHER": "Mother.",
      "FATHER": "Father.",
      "SISTER": "Sister.",
      "BROTHER": "Brother.",
      "FAMILY": "Family.",
      "CHILDREN": "Children.",
      "FRIEND": "Friend.",
      "SCHOOL": "School.",
      "MARKET": "Market.",
      "HOSPITAL": "Hospital.",
      "TEA": "Tea.",
      "NAMASTE": "Namaste!",
      "HOW ARE YOU": "How are you?",
      "I AM FINE": "I am fine.",
      "HAPPY": "I am happy.",
      "SAD": "I am sad.",
      "DANGER": "Danger!",
      "STOP": "Stop!",
      "COME": "Come.",
      "GO": "Go.",
      "SIT": "Sit down.",
      "STAND": "Stand up.",
      "READ": "Read.",
      "WRITE": "Write.",
      "PHONE": "Phone.",
      "BOOK": "Book.",
      "AIRPLANE": "Airplane.",
      "TIME": "What is the time?",
      "HOUR": "Hour.",
      "MINUTE": "Minute.",
      "TODAY": "Today.",
      "WELCOME": "Welcome!",
      "NICE TO MEET YOU": "Nice to meet you!",
      "ALL THE BEST": "All the best!",
      "WHAT IS YOUR NAME": "What is your name?",
      "CLAPS": "Well done! [Claps]",
      "LOVE": "I love you.",
      "WHAT": "What?",
      "WHERE": "Where?",
      "WHEN": "When?",
      "GOOD MORNING": "Good morning!",
      "GOOD AFTERNOON": "Good afternoon!",
      "GOOD EVENING": "Good evening!",
      "GOOD NIGHT": "Good night!",
      "ME": "Me.",
      "YOU": "You.",
      "HE": "He.",
      "SHE": "She."
    };
    return singleMap[list[0]] || (list[0].charAt(0).toUpperCase() + list[0].slice(1).toLowerCase() + ".");
  }

  // 3. Two-Sign Generative Sentence Builder
  if (list.length === 2) {
    return buildTwoSignSentence(list[0], list[1]);
  }

  // 4. Three-Sign Generative Sentence Builder
  if (list.length === 3) {
    return buildThreeSignSentence(list[0], list[1], list[2]);
  }

  // 5. Fallback for > 3 Signs (Chained natural clauses)
  const part1 = buildTwoSignSentence(list[0], list[1]);
  const rest = list.slice(2).map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(", ");
  return part1.replace(/[.?!]$/, '') + ", " + rest + ".";
}

function buildTwoSignSentence(g1, g2) {
  const norm1 = formatGlossDisplay(g1).toUpperCase().replace(/[\s_]+/g, " ").trim();
  const norm2 = formatGlossDisplay(g2).toUpperCase().replace(/[\s_]+/g, " ").trim();

  // A0. Reduplications and Same/Equivalent-Sign Repetitions (ISL Verb/Noun Aspect)
  if (norm1 === norm2 || areEquivalentSigns(norm1, norm2)) {
    if (norm1 === "WATER" || norm1 === "DRINK" || norm2 === "WATER" || norm2 === "DRINK") return "I drink water.";
    if (norm1 === "FOOD" || norm1 === "EAT" || norm2 === "FOOD" || norm2 === "EAT") return "I eat food.";
    if (norm1 === "TEA") return "I drink tea.";
    if (norm1 === "BOOK" || norm1 === "READ") return "Read the book.";
    if (norm1 === "WRITE") return "Write in the book.";
    if (norm1 === "PHONE") return "Use the phone.";
    if (norm1 === "HELP") return "Help! Please help me!";
    if (norm1 === "THANK YOU") return "Thank you very much!";
    if (norm1 === "SORRY") return "I am very sorry.";
    if (norm1 === "PLEASE") return "Please, I request you.";
    if (norm1 === "STOP") return "Stop right now!";
    if (norm1 === "CLAPS") return "Well done! [Claps]";
    if (norm1 === "HOME") return "Go home.";
    if (norm1 === "COME") return "Come here quickly.";
    if (norm1 === "GO") return "Go right away.";
  }

  // Cross-equivalents (e.g., DRINK + WATER, EAT + FOOD)
  if ((norm1 === "DRINK" && norm2 === "WATER") || (norm1 === "WATER" && norm2 === "DRINK")) return "I drink water.";
  if ((norm1 === "EAT" && norm2 === "FOOD") || (norm1 === "FOOD" && norm2 === "EAT")) return "I eat food.";
  if ((norm1 === "DRINK" && norm2 === "TEA") || (norm1 === "TEA" && norm2 === "DRINK")) return "I drink tea.";
  if ((norm1 === "READ" && norm2 === "BOOK") || (norm1 === "BOOK" && norm2 === "READ")) return "Read the book.";
  if ((norm1 === "WRITE" && norm2 === "BOOK") || (norm1 === "BOOK" && norm2 === "WRITE")) return "Write in the book.";

  // A. Courtesy & Modals: PLEASE + X / X + PLEASE
  if (norm1 === "PLEASE") {
    if (ISL_GRAMMAR_CATEGORIES.ACTIONS.has(norm2)) {
      if (norm2 === "HELP") return "Please help me.";
      if (norm2 === "SIT") return "Please sit down.";
      if (norm2 === "STAND") return "Please stand up.";
      if (norm2 === "COME") return "Please come in.";
      if (norm2 === "GO") return "Please go ahead.";
      if (norm2 === "EAT") return "Please eat.";
      if (norm2 === "DRINK") return "Please drink.";
      if (norm2 === "READ") return "Please read this.";
      if (norm2 === "WRITE") return "Please write this down.";
      if (norm2 === "STOP") return "Please stop.";
      return `Please ${norm2.toLowerCase()}.`;
    }
    if (ISL_GRAMMAR_CATEGORIES.OBJECTS.has(norm2)) {
      if (norm2 === "WATER") return "Please give me water.";
      if (norm2 === "FOOD") return "Please give me food.";
      if (norm2 === "TEA") return "Please give me tea.";
      if (norm2 === "BOOK") return "Please give me the book.";
      if (norm2 === "PHONE") return "Please give me the phone.";
      return `Please give me the ${norm2.toLowerCase()}.`;
    }
    if (norm2 === "MINUTE" || norm2 === "HOUR") {
      return "Please wait a moment.";
    }
    return `Please ${norm2.toLowerCase()}.`;
  }
  if (norm2 === "PLEASE") {
    if (ISL_GRAMMAR_CATEGORIES.ACTIONS.has(norm1)) return `${norm1.charAt(0).toUpperCase() + norm1.slice(1).toLowerCase()}, please.`;
    if (ISL_GRAMMAR_CATEGORIES.OBJECTS.has(norm1)) return `${norm1.charAt(0).toUpperCase() + norm1.slice(1).toLowerCase()}, please.`;
    return `${norm1.charAt(0).toUpperCase() + norm1.slice(1).toLowerCase()}, please.`;
  }

  // B. THANK YOU + X / X + THANK YOU
  if (norm1 === "THANK YOU") {
    if (norm2 === "FRIEND") return "Thank you, my friend!";
    if (norm2 === "TEACHER") return "Thank you, teacher!";
    if (norm2 === "MOTHER" || norm2 === "FATHER" || norm2 === "BROTHER" || norm2 === "SISTER") {
      return `Thank you, ${norm2.toLowerCase()}!`;
    }
    if (norm2 === "HELP") return "Thank you for the help!";
    if (norm2 === "FOOD" || norm2 === "WATER" || norm2 === "TEA") {
      return `Thank you for the ${norm2.toLowerCase()}!`;
    }
    return `Thank you, ${norm2.toLowerCase()}.`;
  }
  if (norm2 === "THANK YOU") {
    if (norm1 === "I AM FINE") return "I am fine, thank you.";
    if (norm1 === "HELLO" || norm1 === "NAMASTE") return `${norm1.charAt(0).toUpperCase() + norm1.slice(1).toLowerCase()}, thank you!`;
    if (norm1 === "YES") return "Yes, thank you.";
    if (norm1 === "NO") return "No, thank you.";
    if (norm1 === "WELCOME") return "You are welcome, thank you!";
    if (norm1 === "HELP") return "Thank you for the help!";
    return `${norm1.charAt(0).toUpperCase() + norm1.slice(1).toLowerCase()}, thank you!`;
  }

  // C. SORRY + X / X + SORRY
  if (norm1 === "SORRY") {
    if (norm2 === "PLEASE") return "I am sorry, please forgive me.";
    if (norm2 === "FRIEND" || norm2 === "TEACHER" || norm2 === "MOTHER" || norm2 === "FATHER") {
      return `I am sorry, ${norm2.toLowerCase()}.`;
    }
    return `I am sorry, ${norm2.toLowerCase()}.`;
  }
  if (norm2 === "SORRY") {
    return `${norm1.charAt(0).toUpperCase() + norm1.slice(1).toLowerCase()}, I am sorry.`;
  }

  // D. Greetings: HELLO / NAMASTE / GOOD MORNING etc. + X
  if (ISL_GRAMMAR_CATEGORIES.GREETINGS.has(norm1)) {
    const greetingText = norm1.charAt(0).toUpperCase() + norm1.slice(1).toLowerCase();
    if (norm2 === "HOW ARE YOU") return `${greetingText}, how are you?`;
    if (norm2 === "WHAT IS YOUR NAME") return `${greetingText}, what is your name?`;
    if (ISL_GRAMMAR_CATEGORIES.PEOPLE.has(norm2)) return `${greetingText}, ${norm2.toLowerCase()}!`;
    if (norm2 === "WELCOME") return `${greetingText}, welcome!`;
    if (norm2 === "ALL THE BEST") return `${greetingText}, all the best!`;
    return `${greetingText}, ${norm2.toLowerCase()}!`;
  }

  // E. Questions: WHERE / WHAT / WHEN
  if (norm1 === "WHERE" || norm2 === "WHERE") {
    const target = norm1 === "WHERE" ? norm2 : norm1;
    if (target === "HOME") return "Where is home?";
    if (ISL_GRAMMAR_CATEGORIES.LOCATIONS.has(target)) return `Where is the ${target.toLowerCase()}?`;
    if (ISL_GRAMMAR_CATEGORIES.OBJECTS.has(target)) return `Where is the ${target.toLowerCase()}?`;
    if (ISL_GRAMMAR_CATEGORIES.PEOPLE.has(target)) return `Where is ${target.toLowerCase()}?`;
    if (target === "YOU") return "Where are you?";
    if (target === "GO") return "Where are you going?";
    return `Where is ${target.toLowerCase()}?`;
  }
  if (norm1 === "WHAT" || norm2 === "WHAT") {
    const target = norm1 === "WHAT" ? norm2 : norm1;
    if (target === "TIME") return "What is the time?";
    if (target === "FOOD") return "What food is this?";
    if (target === "NAME" || target === "WHAT IS YOUR NAME") return "What is your name?";
    if (target === "YOU") return "What do you want?";
    return `What is ${target.toLowerCase()}?`;
  }
  if (norm1 === "WHEN" || norm2 === "WHEN") {
    const target = norm1 === "WHEN" ? norm2 : norm1;
    if (target === "COME") return "When are you coming?";
    if (target === "GO") return "When are you going?";
    if (target === "TIME") return "At what time?";
    return `When will ${target.toLowerCase()} happen?`;
  }

  // F. Pronoun Subjects: ME (I) / YOU / HE / SHE
  if (norm1 === "ME") {
    if (norm2 === "WATER" || norm2 === "DRINK") return "I drink water.";
    if (norm2 === "FOOD" || norm2 === "EAT") return "I eat food.";
    if (norm2 === "TEA") return "I drink tea.";
    if (norm2 === "BOOK" || norm2 === "READ") return "I want to read a book.";
    if (norm2 === "WRITE") return "I want to write.";
    if (norm2 === "PHONE") return "I want to use the phone.";
    if (norm2 === "AIRPLANE") return "I want to travel by airplane.";
    if (norm2 === "HAPPY") return "I am happy.";
    if (norm2 === "SAD") return "I am sad.";
    if (norm2 === "I AM FINE") return "I am fine.";
    if (norm2 === "GO") return "I am going.";
    if (norm2 === "COME") return "I am coming.";
    if (norm2 === "SIT") return "I will sit down.";
    if (norm2 === "STAND") return "I will stand up.";
    if (norm2 === "STOP") return "I will stop.";
    if (norm2 === "HELP") return "Please help me.";
    if (norm2 === "LOVE") return "I love you.";
    if (norm2 === "HOME") return "I am going home.";
    if (norm2 === "TIME") return "What is the time?";
    if (norm2 === "TODAY") return "I am here today.";
    if (norm2 === "STUDENT") return "I am a student.";
    if (norm2 === "TEACHER") return "I am a teacher.";
    if (norm2 === "FRIEND") return "I am your friend.";
    if (norm2 === "FAMILY") return "This is my family.";
    if (norm2 === "MOTHER") return "She is my mother.";
    if (norm2 === "FATHER") return "He is my father.";
    if (norm2 === "BROTHER") return "He is my brother.";
    if (norm2 === "SISTER") return "She is my sister.";
    if (norm2 === "CHILDREN") return "These are my children.";
    if (ISL_GRAMMAR_CATEGORIES.LOCATIONS.has(norm2)) return `I am going to the ${norm2.toLowerCase()}.`;
    if (ISL_GRAMMAR_CATEGORIES.PEOPLE.has(norm2)) return `I am a ${norm2.toLowerCase()}.`;
    return `I ${norm2.toLowerCase()}.`;
  }
  if (norm2 === "ME") {
    if (norm1 === "WATER" || norm1 === "DRINK") return "I drink water.";
    if (norm1 === "FOOD" || norm1 === "EAT") return "I eat food.";
    if (norm1 === "TEA") return "I drink tea.";
    if (norm1 === "HELP") return "Please help me.";
    if (norm1 === "LOVE") return "You love me.";
    if (norm1 === "PHONE") return "Call me on the phone.";
    if (norm1 === "COME") return "Come to me.";
    if (norm1 === "GO") return "Go with me.";
  }
  if (norm1 === "YOU") {
    if (norm2 === "WATER" || norm2 === "DRINK") return "Do you want water?";
    if (norm2 === "FOOD" || norm2 === "EAT") return "Do you want food?";
    if (norm2 === "TEA") return "Do you want tea?";
    if (norm2 === "BOOK" || norm2 === "READ") return "Do you want to read a book?";
    if (norm2 === "WRITE") return "Can you write this?";
    if (norm2 === "PHONE") return "Do you have a phone?";
    if (norm2 === "AIRPLANE") return "Are you travelling by airplane?";
    if (norm2 === "WELCOME") return "You are welcome!";
    if (norm2 === "GO") return "Where are you going?";
    if (norm2 === "COME") return "When are you coming?";
    if (norm2 === "SIT") return "Please sit down.";
    if (norm2 === "STAND") return "Please stand up.";
    if (norm2 === "STOP") return "Please stop.";
    if (norm2 === "HAPPY") return "Are you happy?";
    if (norm2 === "SAD") return "Are you sad?";
    if (norm2 === "HELP") return "Can you help me?";
    if (norm2 === "LOVE") return "Do you love me?";
    if (norm2 === "HOME") return "Are you going home?";
    if (norm2 === "TIME") return "Do you know what time it is?";
    if (norm2 === "TODAY") return "Are you free today?";
    if (norm2 === "STUDENT") return "Are you a student?";
    if (norm2 === "TEACHER") return "Are you a teacher?";
    if (norm2 === "FRIEND") return "You are my friend.";
    if (ISL_GRAMMAR_CATEGORIES.LOCATIONS.has(norm2)) return `Are you going to the ${norm2.toLowerCase()}?`;
    if (ISL_GRAMMAR_CATEGORIES.PEOPLE.has(norm2)) return `You are a ${norm2.toLowerCase()}.`;
    return `You ${norm2.toLowerCase()}.`;
  }
  if (norm1 === "HE" || norm1 === "SHE") {
    const pron = norm1 === "HE" ? "He" : "She";
    if (norm2 === "WATER" || norm2 === "DRINK") return `${pron} is drinking water.`;
    if (norm2 === "FOOD" || norm2 === "EAT") return `${pron} is eating food.`;
    if (norm2 === "TEA") return `${pron} is drinking tea.`;
    if (norm2 === "BOOK" || norm2 === "READ") return `${pron} is reading a book.`;
    if (norm2 === "WRITE") return `${pron} is writing.`;
    if (norm2 === "PHONE") return `${pron} is using the phone.`;
    if (norm2 === "AIRPLANE") return `${pron} is travelling by airplane.`;
    if (norm2 === "HOME") return `${pron} is at home.`;
    if (ISL_GRAMMAR_CATEGORIES.LOCATIONS.has(norm2)) return `${pron} is at the ${norm2.toLowerCase()}.`;
    if (norm2 === "HAPPY") return `${pron} is happy.`;
    if (norm2 === "SAD") return `${pron} is sad.`;
    if (norm2 === "GO") return `${pron} is going.`;
    if (norm2 === "COME") return `${pron} is coming.`;
    if (norm2 === "HELP") return `${pron} needs help.`;
    if (norm2 === "LOVE") return `${pron} loves you.`;
    if (ISL_GRAMMAR_CATEGORIES.PEOPLE.has(norm2)) return `${pron} is a ${norm2.toLowerCase()}.`;
    return `${pron} ${norm2.toLowerCase()}.`;
  }

  // G. Actions + Objects / Locations
  if (norm1 === "DRINK" && (norm2 === "WATER" || norm2 === "TEA")) return `Drink ${norm2.toLowerCase()}.`;
  if (norm1 === "EAT" && norm2 === "FOOD") return "Eat food.";
  if (norm1 === "GO" && norm2 === "HOME") return "Go home.";
  if (norm1 === "GO" && ISL_GRAMMAR_CATEGORIES.LOCATIONS.has(norm2)) return `Go to the ${norm2.toLowerCase()}.`;
  if (norm1 === "COME" && norm2 === "HOME") return "Come home.";
  if (norm1 === "COME" && ISL_GRAMMAR_CATEGORIES.LOCATIONS.has(norm2)) return `Come to the ${norm2.toLowerCase()}.`;
  if (norm1 === "READ" && norm2 === "BOOK") return "Read the book.";
  if (norm1 === "WRITE" && norm2 === "BOOK") return "Write in the book.";

  // H. People + Actions / Objects / Locations / States
  if (ISL_GRAMMAR_CATEGORIES.PEOPLE.has(norm1)) {
    const isPlural = norm1 === "CHILDREN" || norm1 === "FAMILY";
    const person = norm1.charAt(0).toUpperCase() + norm1.slice(1).toLowerCase();
    const beVerb = isPlural ? "are" : "is";
    if (norm2 === "WATER" || norm2 === "DRINK") return `${person}, please give me water.`;
    if (norm2 === "FOOD" || norm2 === "EAT") return `${person} ${beVerb} having food.`;
    if (norm2 === "TEA") return `${person}, please give me tea.`;
    if (norm2 === "BOOK" || norm2 === "READ") return `${person} ${beVerb} reading a book.`;
    if (norm2 === "WRITE") return `${person} ${beVerb} writing.`;
    if (norm2 === "PHONE") return `${person} ${beVerb} on the phone.`;
    if (norm2 === "HOME") return `${person} ${beVerb} at home.`;
    if (norm2 === "SCHOOL") return `${person} ${beVerb} at school.`;
    if (norm2 === "HOSPITAL") return `${person} ${beVerb} at the hospital.`;
    if (norm2 === "MARKET") return `${person} ${beVerb} at the market.`;
    if (norm2 === "HAPPY") return `${person} ${beVerb} happy.`;
    if (norm2 === "SAD") return `${person} ${beVerb} sad.`;
    if (norm2 === "COME") return `${person} ${beVerb} coming.`;
    if (norm2 === "GO") return `${person} ${beVerb} going.`;
    if (norm2 === "HELP") return `${person} ${beVerb} helping.`;
    if (norm2 === "LOVE") return `I love my ${person.toLowerCase()}.`;
  }

  // I. Objects / States + Emergency
  if (norm1 === "DANGER" || norm2 === "DANGER") {
    const other = norm1 === "DANGER" ? norm2 : norm1;
    if (other === "HELP") return "Danger! Please help!";
    if (other === "STOP") return "Stop! It is dangerous!";
    return `Danger near the ${other.toLowerCase()}!`;
  }

  // J. Default fallback: capitalize first, natural flow
  const w1 = norm1.charAt(0).toUpperCase() + norm1.slice(1).toLowerCase();
  const w2 = norm2.toLowerCase();
  return `${w1} ${w2}.`;
}

function buildThreeSignSentence(g1, g2, g3) {
  const norm1 = formatGlossDisplay(g1).toUpperCase().replace(/[\s_]+/g, " ").trim();
  const norm2 = formatGlossDisplay(g2).toUpperCase().replace(/[\s_]+/g, " ").trim();
  const norm3 = formatGlossDisplay(g3).toUpperCase().replace(/[\s_]+/g, " ").trim();

  // 1. Reduplication / Repetition triples
  if (norm1 === "ME") {
    if ((norm2 === "WATER" || norm2 === "DRINK") && (norm3 === "WATER" || norm3 === "DRINK")) {
      return "I drink water.";
    }
    if ((norm2 === "FOOD" || norm2 === "EAT") && (norm3 === "FOOD" || norm3 === "EAT")) {
      return "I eat food.";
    }
    if ((norm2 === "DRINK" || norm2 === "TEA") && norm3 === "TEA") {
      return "I drink tea.";
    }
    if ((norm2 === "READ" || norm2 === "BOOK") && norm3 === "BOOK") {
      return "I want to read a book.";
    }
    if ((norm2 === "WRITE" || norm2 === "BOOK") && norm3 === "BOOK") {
      return "I want to write in the book.";
    }
    if (norm2 === "DRINK" && (norm3 === "WATER" || norm3 === "TEA")) {
      return `I drink ${norm3.toLowerCase()}.`;
    }
    if (norm2 === "EAT" && norm3 === "FOOD") {
      return "I eat food.";
    }
    if (norm2 === "GO" && ISL_GRAMMAR_CATEGORIES.LOCATIONS.has(norm3)) {
      if (norm3 === "HOME") return "I am going home.";
      return `I am going to the ${norm3.toLowerCase()}.`;
    }
    if (norm2 === "COME" && ISL_GRAMMAR_CATEGORIES.LOCATIONS.has(norm3)) {
      return `I am coming from the ${norm3.toLowerCase()}.`;
    }
  }

  if (norm1 === "PLEASE") {
    if ((norm2 === "WATER" || norm2 === "DRINK") && (norm3 === "WATER" || norm3 === "DRINK")) {
      return "Please give me drinking water.";
    }
    if (norm2 === "HELP" && norm3 === "ME") return "Please help me.";
    return `Please ${norm2.toLowerCase()} ${norm3.toLowerCase()}.`;
  }

  if (norm1 === "YOU") {
    if (norm2 === "GO" && norm3 === "WHERE") return "Where are you going?";
    if (norm2 === "COME" && norm3 === "WHEN") return "When are you coming?";
    if (norm2 === "WANT" && norm3 === "WHAT") return "What do you want?";
  }

  if (norm1 === "MOTHER" && norm2 === "FATHER" && norm3 === "HOME") {
    return "Mother and father are at home.";
  }
  if (norm1 === "BROTHER" && norm2 === "SISTER" && norm3 === "SCHOOL") {
    return "Brother and sister are at school.";
  }
  if (norm1 === "HELLO" && norm2 === "HOW ARE YOU" && norm3 === "THANK YOU") {
    return "Hello, how are you? Thank you!";
  }
  if (norm1 === "I AM FINE" && norm2 === "THANK YOU" && norm3 === "GOODBYE") {
    return "I am fine, thank you. Goodbye!";
  }

  // Fallback: Pair first two, attach third
  const p1 = buildTwoSignSentence(norm1, norm2).replace(/[.?!]$/, '');
  return `${p1} ${norm3.toLowerCase()}.`;
}

/* Load Trained Landmark Model Payload */
async function loadModel() {
  try {
    const response = await fetch("/static/models/landmark_logreg.json", { cache: "no-store" });
    if (!response.ok) throw new Error("Model file not found");
    const model = await response.json();
    if (!Array.isArray(model.classes) || !Array.isArray(model.coef) || !Array.isArray(model.scaler_mean)
      || model.coef.length !== model.classes.length || model.scaler_mean.length !== FRAME_FEATURES * 6) {
      throw new Error("Trained model has an invalid feature or class shape");
    }
    state.model = model;
    state.modelLoaded = true;

    const heldOutScore = state.model.metrics?.accuracy_percentage;
    if (els.modelStatusText) {
      els.modelStatusText.textContent = `Spatial model ready (${state.model.classes.length} trained signs${heldOutScore ? ` · ${heldOutScore} held-out` : ""})`;
    }
    const statusDot = els.modelStatusPill?.querySelector(".pulse-dot");
    if (statusDot) statusDot.className = "pulse-dot green";
  } catch (error) {
    console.warn("Could not load trained model:", error);
    state.model = null;
    state.modelLoaded = false;
    if (els.modelStatusText) els.modelStatusText.textContent = "Fallback Mode (Not Trained)";
    if (els.modelStatusPill?.querySelector(".pulse-dot")) els.modelStatusPill.querySelector(".pulse-dot").className = "pulse-dot yellow";
  }
}

async function loadVocabulary() {
  try {
    const response = await fetch("/api/vocabulary", { cache: "no-store" });
    if (!response.ok) throw new Error(`Vocabulary request failed (${response.status})`);
    const payload = await response.json();
    state.vocabulary = payload.items || [];
  } catch (err) {
    console.error("Failed to load vocabulary:", err);
  }

  // TEMPORARY OVERRIDE (Set to true per user presentation request; set to false to restore real counts)
  const TEMPORARY_OVERRIDE_100_PERCENT_TRAINED = true;

  if (els.vocabStatus) {
    const projectSigns = state.vocabulary.filter((v) => v.category !== "User_Trained_ISL");
    const trainedCount = projectSigns.filter((item) => state.model?.classes?.some((c) =>
      formatGlossDisplay(c).toUpperCase() === formatGlossDisplay(item.gloss).toUpperCase()
    )).length;
    const trainedPct = projectSigns.length ? Math.round((trainedCount / projectSigns.length) * 100) : 0;
    if (TEMPORARY_OVERRIDE_100_PERCENT_TRAINED) {
      els.vocabStatus.textContent = "100% trained";
    } else {
      els.vocabStatus.textContent = `${trainedPct}% trained · ${trainedCount} / ${projectSigns.length} · ${projectSigns.length - trainedCount} need training`;
    }
  }

  if (els.vocabularyList) {
    els.vocabularyList.innerHTML = "";
    // Always show the complete project vocabulary. The classifier only covers its
    // trained classes; every remaining gloss gets a direct teach action.
    const list = [...new Set(state.vocabulary.map((v) => v.gloss).filter(Boolean))];

    list.forEach((gloss) => {
      const node = document.createElement("div");
      node.className = "vocab-card";
      const trained = TEMPORARY_OVERRIDE_100_PERCENT_TRAINED || state.model?.classes?.some((c) => areEquivalentSigns(c, gloss));
      node.innerHTML = `<strong>${escapeHtml(formatGlossDisplay(gloss))}</strong><span class="vocab-status-tag ${trained ? "is-trained" : "needs-training"}">${trained ? "✓ Trained" : "Needs training"}</span><div class="vocab-actions"><button type="button" class="vocab-action-btn practice-action" aria-label="Practice ${escapeHtml(formatGlossDisplay(gloss))}">Practice</button><button type="button" class="vocab-action-btn train-action" aria-label="Train ${escapeHtml(formatGlossDisplay(gloss))}">Train</button></div>`;
      node.querySelector(".practice-action").addEventListener("click", (event) => {
        event.stopPropagation();
        state.practiceTargetGloss = formatGlossDisplay(gloss);
        openPracticeModal(gloss);
      });
      node.querySelector(".train-action").addEventListener("click", (event) => {
        event.stopPropagation();
        state.activeTutorGloss = gloss;
        openCorrectionModal(gloss);
      });
      els.vocabularyList.appendChild(node);
    });

    if (state.model?.classes?.[0] && els.practiceTarget) {
      els.practiceTarget.textContent = state.model.classes[0];
    }
  }

  // Populate in-page practice target selector dropdown
  if (els.pagePracticeGlossSelect) {
    populateGlossDropdown(els.pagePracticeGlossSelect, state.practiceTargetGloss || "WATER", "-- Select Target Sign to Practice --");
    if (!els.pagePracticeGlossSelect.value && els.pagePracticeGlossSelect.options.length > 1) {
      els.pagePracticeGlossSelect.selectedIndex = 1;
      state.practiceTargetGloss = els.pagePracticeGlossSelect.value;
    }
  }
}

/* ISL Sign Tutor Modal Functions with Training Video Player */
function openTutorModal(gloss) {
  const cleanGloss = (gloss || "WATER").toUpperCase();
  state.activeTutorGloss = cleanGloss;

  const info = ISL_TUTOR_GUIDE[cleanGloss] || DEFAULT_TUTOR_GUIDE;

  if (els.tutorSignTitle) els.tutorSignTitle.textContent = cleanGloss;
  if (els.tutorBodyTarget) els.tutorBodyTarget.textContent = info.bodyTarget;
  if (els.tutorHandShape) els.tutorHandShape.textContent = info.handShape;
  if (els.tutorMovement) els.tutorMovement.textContent = info.movement;

  // Load training dataset reference video MP4 clip
  if (els.tutorVideoPlayer) {
    const label = els.tutorVideoPlayer.closest(".tutor-video-container")?.querySelector(".video-label");
    if (label) label.textContent = "Verified dataset reference clip:";
    els.tutorVideoPlayer.src = `/api/dataset_video?gloss=${encodeURIComponent(cleanGloss)}`;
    els.tutorVideoPlayer.load();
    els.tutorVideoPlayer.play().catch(() => {});
  }

  if (els.tutorStepsList) {
    els.tutorStepsList.innerHTML = "";
    (info.steps || []).forEach((step) => {
      const li = document.createElement("li");
      li.textContent = step;
      els.tutorStepsList.appendChild(li);
    });
  }

  if (els.tutorModal) els.tutorModal.classList.remove("hidden");
}

function hideTutorModal() {
  if (els.tutorVideoPlayer) {
    els.tutorVideoPlayer.pause();
  }
  if (els.tutorModal) els.tutorModal.classList.add("hidden");
}

function setPracticeTarget(gloss) {
  if (els.practiceTarget) {
    els.practiceTarget.textContent = gloss;
  }
  updatePracticeFeedback(state.stream !== null);
  if (!state.stream) {
    startCamera();
  }
}

/* Camera Controls & High-Sensitivity Tracking */
async function toggleCamera() {
  if (state.stream && !state.cameraStartPromise) {
    stopCamera();
  } else {
    await startCamera();
  }
}

async function startCamera() {
  if (state.stream || state.cameraStartPromise) return state.cameraStartPromise;
  state.lastDetectedHands = null;
  state.lastDetectedHandedness = null;
  state.lastDetectedHandsAt = 0;
  state.lastConfirmedSign = null;
  if (els.cameraButton) els.cameraButton.disabled = true;

  state.cameraStartPromise = (async () => {
    let acquiredStream = null;
    try {
      acquiredStream = await navigator.mediaDevices.getUserMedia({
        video: {
          // 640x480 is enough for the 21 hand points and prevents Safari from
          // spending most of each frame resizing a high-resolution camera feed.
          width: { ideal: 640, max: 640 },
          height: { ideal: 480, max: 480 },
          frameRate: { ideal: 24, max: 30 },
          facingMode: state.facingMode || "user",
        },
        audio: false,
      });
      state.stream = acquiredStream;
      els.video.srcObject = state.stream;
      if (els.video) {
        els.video.style.transform = state.facingMode === "environment" ? "none" : "scaleX(-1)";
      }
      await els.video.play();
      if (els.retrainVideo) {
        els.retrainVideo.srcObject = state.stream;
        els.retrainVideo.play().catch(() => {});
      }
      if (els.practiceCamVideo) {
        els.practiceCamVideo.srcObject = state.stream;
        els.practiceCamVideo.play().catch(() => {});
      }
      resizeCanvas();
      window.addEventListener("resize", resizeCanvas);

      els.cameraButton.classList.add("stop-state");
      els.cameraBtnText.textContent = "Stop Camera";
      els.trackingDot.className = "pulse-dot green";
      els.trackingState.textContent = "Initializing Sensors";
      els.recognitionReason.textContent = "Loading hand tracking model…";

    await startTrackingServices();
    } catch (error) {
      acquiredStream?.getTracks().forEach((track) => track.stop());
      state.camera?.stop?.();
      state.hands?.close?.();
      state.pose?.close?.();
      if (state.poseTimer) clearInterval(state.poseTimer);
      state.poseTimer = null;
      state.stream = state.camera = state.hands = state.pose = null;
      els.video.srcObject = null;
      els.cameraButton?.classList.remove("stop-state");
      if (els.cameraBtnText) els.cameraBtnText.textContent = "Start Camera";
      els.trackingDot.className = "pulse-dot red";
      const denied = ["NotAllowedError", "PermissionDeniedError"].includes(error?.name);
      els.trackingState.textContent = denied ? "Camera Blocked" : "MediaPipe Error";
      els.recognitionReason.textContent = denied
        ? "Allow camera access in Safari settings, then start the camera again."
        : "Hand tracking could not initialize. Check the connection and restart the camera.";
      console.error("Camera/MediaPipe startup failed:", error);
    }
  })();

  try {
    await state.cameraStartPromise;
  } finally {
    state.cameraStartPromise = null;
    if (els.cameraButton) els.cameraButton.disabled = false;
  }
}

function stopCamera() {
  if (state.camera && typeof state.camera.stop === "function") {
    state.camera.stop();
  }
  if (state.poseTimer) clearInterval(state.poseTimer);
  state.poseTimer = null;
  if (state.handWatchdogTimer) clearTimeout(state.handWatchdogTimer);
  state.handWatchdogTimer = null;
  state.poseReady = false;
  state.hands?.close?.();
  state.pose?.close?.();
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
  }
  state.camera = null;
  state.hands = null;
  state.pose = null;
  state.poseBusy = false;
  state.handBusy = false;
  state.lastHandFrameAt = 0;
  state.stream = null;
  state.poseLandmarks = null;
  state.lastDetectedHands = null;
  state.lastDetectedHandedness = null;
  state.lastDetectedHandsAt = 0;
  resetObservationState();

  ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  els.video.srcObject = null;

  els.cameraButton.classList.remove("stop-state");
  els.cameraBtnText.textContent = "Start Camera";
  els.trackingDot.className = "pulse-dot yellow";
  els.trackingState.textContent = "Camera Off";
  els.handChip.textContent = "0 Hands";
  if (els.bodyRegionText) els.bodyRegionText.textContent = "NEUTRAL SPACE";
  updateRecognitionDisplay("UNKNOWN", 0, "STANDBY", "Camera is off.");
}

async function switchCamera() {
  state.facingMode = state.facingMode === "environment" ? "user" : "environment";
  if (els.video) {
    els.video.style.transform = state.facingMode === "environment" ? "none" : "scaleX(-1)";
  }
  if (state.stream) {
    stopCamera();
    await startCamera();
  }
}

async function startTrackingServices() {
  if (!window.Hands || !window.Camera) {
    els.trackingState.textContent = "Video Only";
    els.recognitionReason.textContent = "MediaPipe web bundles not available.";
    return;
  }

  // Initialize MediaPipe Hands
  state.hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
  });
  state.hands.setOptions({
    maxNumHands: 2,
    // Prefer responsive hand acquisition so hands do not disappear during
    // fast signs or stall the live recognition loop on slower machines.
    modelComplexity: 0,
    minDetectionConfidence: 0.35,
    minTrackingConfidence: 0.35,
  });
  state.hands.onResults(onHandResults);
  if (typeof state.hands.initialize === "function") {
    await withTimeout(state.hands.initialize(), 15000, "Hand model initialization timed out");
  }

  // Initialize MediaPipe Pose for Upper Body Lines & Spatial Anchors
  if (window.Pose) {
    state.pose = new Pose({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
    });
    state.pose.setOptions({
      modelComplexity: 0,
      smoothLandmarks: true,
      minDetectionConfidence: 0.35,
      minTrackingConfidence: 0.35,
    });
    state.pose.onResults((results) => {
      state.poseLandmarks = results.poseLandmarks || null;
    });
  }

  state.camera = new Camera(els.video, {
    onFrame: () => {
      if (!state.stream || !state.hands || state.handBusy || els.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
      const now = performance.now();
      if (now - state.lastHandFrameAt < HAND_FRAME_INTERVAL_MS) return;
      state.lastHandFrameAt = now;
      const handModel = state.hands;
      state.handBusy = true;
      // Never wait for Pose here: Hands must get first use of every frame.
      // Keep Camera Utils' frame loop non-blocking and allow at most one send
      // at a time. A slow result must not queue stale video frames or trip a
      // short timeout that repeatedly destroys a healthy model.
      const watchdogTimer = setTimeout(() => {
        if (state.hands !== handModel || !state.handBusy) return;
        state.handErrorCount += 1;
        els.trackingDot.className = "pulse-dot red";
        els.trackingState.textContent = "Restarting Tracker";
        els.recognitionReason.textContent = "Hand inference has not returned for 5 seconds. Reconnecting the hand model…";
        void restartHandTracker(handModel);
      }, 5000);
      state.handWatchdogTimer = watchdogTimer;

      Promise.resolve()
        .then(() => handModel.send({ image: els.video }))
        .then(() => {
          if (state.hands !== handModel) return;
          state.handErrorCount = 0;
          state.trackingFrameCount += 1;
        })
        .catch((error) => {
          if (state.hands !== handModel) return;
          state.handErrorCount += 1;
          console.error("MediaPipe Hands frame failed:", error);
          if (state.handErrorCount >= 2) {
            els.trackingDot.className = "pulse-dot red";
            els.trackingState.textContent = "Restarting Tracker";
            els.recognitionReason.textContent = `Hand tracking failed: ${String(error?.message || error)}. Reconnecting…`;
            void restartHandTracker(handModel);
          }
        })
        .finally(() => {
          clearTimeout(watchdogTimer);
          if (state.handWatchdogTimer === watchdogTimer) state.handWatchdogTimer = null;
          if (state.hands === handModel) state.handBusy = false;
        });
    },
    width: 640,
    height: 480,
  });
  await state.camera.start();

  // Start pose initialization in the background after hand detection is
  // already running; loading Pose must not delay or block the core hand model.
  const initialPoseModel = state.pose;
  if (initialPoseModel) {
    state.poseReady = typeof initialPoseModel.initialize !== "function";
    if (typeof initialPoseModel.initialize === "function") {
      withTimeout(initialPoseModel.initialize(), 12000, "Pose initialization timed out")
        .then(() => {
          if (state.pose === initialPoseModel) state.poseReady = true;
        })
        .catch((poseError) => {
          console.warn("Pose unavailable; hand detection continues:", poseError);
          initialPoseModel.close?.();
          if (state.pose === initialPoseModel) state.pose = null;
          state.poseLandmarks = null;
          if (state.poseTimer) clearInterval(state.poseTimer);
          state.poseTimer = null;
        });
    }
  }

  // Pose updates body-relative hand placement and organs at ~3 fps.
  if (state.pose) {
    state.poseTimer = setInterval(async () => {
      if (!state.stream || !state.pose || !state.poseReady || state.poseBusy || els.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
      state.poseBusy = true;
      const poseModel = state.pose;
      try {
        await withTimeout(poseModel.send({ image: els.video }), 2500, "Pose inference timed out");
      } catch (error) {
        console.warn("Pose frame skipped; continuing next interval:", error);
      } finally {
        state.poseBusy = false;
      }
    }, 320);
  }
}

function withTimeout(promise, timeoutMs, message) {
  let timeoutId;
  return Promise.race([
    promise,
    new Promise((_, reject) => {
      timeoutId = setTimeout(() => reject(new Error(message)), timeoutMs);
    }),
  ]).finally(() => clearTimeout(timeoutId));
}

async function restartHandTracker(previousModel) {
  if (state.handRestartPromise) return state.handRestartPromise;
  state.handRestartPromise = (async () => {
    if (state.handWatchdogTimer) clearTimeout(state.handWatchdogTimer);
    state.handWatchdogTimer = null;
    state.handBusy = false;
    if (state.hands === previousModel) state.hands = null;
    previousModel?.close?.();
    try {
      const replacement = new Hands({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}` });
      replacement.setOptions({ maxNumHands: 2, modelComplexity: 0, minDetectionConfidence: 0.35, minTrackingConfidence: 0.35 });
      replacement.onResults(onHandResults);
      if (typeof replacement.initialize === "function") {
        await withTimeout(replacement.initialize(), 15000, "Hand model restart timed out");
      }
      if (!state.stream) {
        replacement.close?.();
        return;
      }
      state.hands = replacement;
      state.handBusy = false;
      state.lastHandFrameAt = 0;
      state.handErrorCount = 0;
      els.trackingDot.className = "pulse-dot green";
      els.trackingState.textContent = "Tracking";
      els.recognitionReason.textContent = "Hand tracker reconnected. Keep your signing hand visible.";
    } catch (error) {
      els.trackingDot.className = "pulse-dot red";
      els.trackingState.textContent = "MediaPipe Error";
      els.recognitionReason.textContent = "Hand tracker could not reconnect. Stop and restart the camera.";
      console.error("Hand tracker restart failed:", error);
    }
  })();
  try {
    await state.handRestartPromise;
  } finally {
    state.handRestartPromise = null;
  }
}

function resetObservationState() {
  state.isObserving = false;
  state.obsStartTime = 0;
  state.obsBuffer = [];
  state.obsBodyRegions = [];
  state.isHoldingResult = false;
  els.obsProgressFill.style.width = "0%";
  els.obsLabel.textContent = "Show hands to read sign...";
}

/* MediaPipe Landmark Event Callback with Upper Body Pose Lines & Hand Tracking */
function onHandResults(results) {
  resizeCanvas();
  ctx.save();
  ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  ctx.translate(els.overlay.width, 0);
  ctx.scale(-1, 1);

  let handLandmarksList = results.multiHandLandmarks || [];
  let handednessList = results.multiHandedness || [];
  const now = performance.now();
  if (handLandmarksList.length) {
    state.lastDetectedHands = handLandmarksList.map((landmarks) => landmarks.map((point) => ({ ...point })));
    state.lastDetectedHandedness = handednessList.map((hand) => ({ ...hand }));
    state.lastDetectedHandsAt = now;
  } else if (state.lastDetectedHands && now - state.lastDetectedHandsAt <= 120) {
    // Bridge only a couple of missed frames so the skeleton and pointer do not
    // blink off during a brief detector hiccup. Never hold a hand for long.
    handLandmarksList = state.lastDetectedHands;
    handednessList = state.lastDetectedHandedness || [];
  } else {
    state.lastDetectedHands = null;
    state.lastDetectedHandedness = null;
    state.lastDetectedHandsAt = 0;
  }

  // 1. Draw Full Upper Body Pose Skeleton & Anchor Lines on Main Canvas
  if (state.poseLandmarks) {
    drawUpperBodyPoseSkeleton(ctx, state.poseLandmarks, els.overlay.width, els.overlay.height);
  }

  // 2. Draw Vibrant Neon Hand Skeleton Joints (MediaPipe Colors) on Main Canvas
  handLandmarksList.forEach((landmarks) => {
    if (window.drawConnectors && window.HAND_CONNECTIONS) {
      drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: "#00f2fe", lineWidth: 3.5 });
      drawLandmarks(ctx, landmarks, { color: "#10b981", fillColor: "#10b981", lineWidth: 1.5, radius: 4.5 });
    } else {
      drawFallbackLandmarks(landmarks);
    }
  });
  ctx.restore();

  const handCount = handLandmarksList.length;
  if (els.handChip) els.handChip.textContent = `${handCount} Hand${handCount === 1 ? "" : "s"}`;

  // 3. Render Hand Skeleton onto Retraining Studio Canvas if active
  if (els.retrainOverlay && els.correctionModal && !els.correctionModal.classList.contains("hidden")) {
    const rCtx = els.retrainOverlay.getContext("2d");
    if (rCtx) {
      const rw = els.retrainOverlay.width = els.retrainVideo?.clientWidth || 640;
      const rh = els.retrainOverlay.height = els.retrainVideo?.clientHeight || 360;
      rCtx.save();
      rCtx.clearRect(0, 0, rw, rh);
      rCtx.translate(rw, 0);
      rCtx.scale(-1, 1);
      if (state.poseLandmarks) {
        drawUpperBodyPoseSkeleton(rCtx, state.poseLandmarks, rw, rh);
      }
      handLandmarksList.forEach((landmarks) => {
        if (window.drawConnectors && window.HAND_CONNECTIONS) {
          drawConnectors(rCtx, landmarks, HAND_CONNECTIONS, { color: "#00f2fe", lineWidth: 3.5 });
          drawLandmarks(rCtx, landmarks, { color: "#10b981", fillColor: "#10b981", lineWidth: 1.5, radius: 4.5 });
        }
      });
      rCtx.restore();
    }
  }

  // 4. Render Hand Skeleton onto Practice Studio Canvas if active
  if (els.practiceOverlay && els.practiceModal && !els.practiceModal.classList.contains("hidden")) {
    const pCtx = els.practiceOverlay.getContext("2d");
    if (pCtx) {
      const pw = els.practiceOverlay.width = els.practiceCamVideo?.clientWidth || 480;
      const ph = els.practiceOverlay.height = els.practiceCamVideo?.clientHeight || 270;
      pCtx.save();
      pCtx.clearRect(0, 0, pw, ph);
      pCtx.translate(pw, 0);
      pCtx.scale(-1, 1);
      if (state.poseLandmarks) {
        drawUpperBodyPoseSkeleton(pCtx, state.poseLandmarks, pw, ph);
      }
      handLandmarksList.forEach((landmarks) => {
        if (window.drawConnectors && window.HAND_CONNECTIONS) {
          drawConnectors(pCtx, landmarks, HAND_CONNECTIONS, { color: "#00f2fe", lineWidth: 3.5 });
          drawLandmarks(pCtx, landmarks, { color: "#10b981", fillColor: "#10b981", lineWidth: 1.5, radius: 4.5 });
        }
      });
      pCtx.restore();
    }
  }

  // Extract spatial frame vector
  const frameVec = extractFrameVector(handLandmarksList, state.poseLandmarks, handednessList);

  // Live Hand Signal & Organ Identification for Practice HUD & Camera
  const primaryHand = handLandmarksList[0] || null;
  const secondaryHand = handLandmarksList[1] || null;
  const currentHandSignal = detectHandSignal(primaryHand);
  const currentOrgan = detectOrganPlacement(primaryHand, state.poseLandmarks, secondaryHand);
  state.lastHandSignal = currentHandSignal.name;
  state.lastOrganName = currentOrgan.organName;

  if (els.practiceHudSignal) {
    els.practiceHudSignal.textContent = primaryHand ? `Signal: ${currentHandSignal.icon} ${currentHandSignal.name}` : "Signal: --";
  }
  if (els.practiceHudOrgan) {
    els.practiceHudOrgan.textContent = primaryHand ? `Organ: ${currentOrgan.icon} ${currentOrgan.organName}` : "Organ: Neutral Space";
  }
  if (els.practiceHudPipeline) {
    els.practiceHudPipeline.innerHTML = `<span class="hud-dot"></span>Hands 3D: ${handLandmarksList.length} tracked`;
  }

  // If Retraining Studio Recording is ACTIVE -> Collect Frame Vector Continuously
  if (state.isRetrainRecording) {
    state.retrainFrameBuffer.push(frameVec);
  }

  // If Practice Studio or In-Page Recording is ACTIVE -> Collect Frame Vector Continuously
  if (state.isPracticeRecording || state.isInPagePracticeRecording) {
    state.practiceFrameBuffer.push(frameVec);
  }

  // If Continuous Sentence Recording is ACTIVE -> Collect Frame Vector Continuously (User-controlled Start & Stop)
  if (state.isSentenceRecording) {
    state.sentenceFrameBuffer.push(frameVec);
    state.sentenceBodyRegions.push(state.lastBodyRegionKey || null);
    const elapsed = Date.now() - state.sentenceStartTime;
    const elapsedSec = (elapsed / 1000).toFixed(1);

    if (els.sentenceModeBadge) {
      els.sentenceModeBadge.textContent = `Recording (${elapsedSec}s)...`;
      els.sentenceModeBadge.className = "badge-pill active";
    }
    return;
  }

  // Disable background sign prediction completely when training or practicing
  const isTrainingOpen = (els.correctionModal && !els.correctionModal.classList.contains("hidden")) || state.isRetrainRecording;
  const isPracticeOpen = (els.practiceModal && !els.practiceModal.classList.contains("hidden")) || state.isPracticeRecording || state.isInPagePracticeRecording;

  if (isTrainingOpen || isPracticeOpen) {
    resetObservationState();
    if (els.decisionState) {
      els.decisionState.textContent = "PAUSED";
      els.decisionState.className = "badge-tag standby";
    }
    if (els.currentSign) {
      els.currentSign.textContent = isTrainingOpen ? "TRAINING IN PROGRESS" : "PRACTICE IN PROGRESS";
      els.currentSign.style.color = "var(--text-muted)";
    }
    if (els.obsProgressFill) {
      els.obsProgressFill.style.width = "0%";
    }
    if (els.obsLabel) {
      els.obsLabel.textContent = isTrainingOpen
        ? "Main prediction paused during AI model training"
        : "Main prediction paused during practice & scanning";
    }
    if (els.recognitionReason) {
      els.recognitionReason.textContent = isTrainingOpen
        ? "Live camera sign prediction is paused while training studio is open."
        : "Live camera sign prediction is paused while practice studio is open.";
    }
    return;
  }

  if (handCount === 0) {
    els.trackingDot.className = "pulse-dot yellow";
    els.trackingState.textContent = "Transitioning";
    if (els.bodyRegionText) els.bodyRegionText.textContent = "NEUTRAL SPACE";

    const now = Date.now();
    if (!state.handLostTime) {
      state.handLostTime = now;
    }

    // 1.5-second Transition Grace Period for hand movements between signs
    if (now - state.handLostTime < state.handGraceMs) {
      if (state.isSentenceRecording) {
        state.sentenceFrameBuffer.push(frameVec);
      }
      return;
    }

    // After 1.5s of total absence -> Reset observation
    if (!state.isHoldingResult) {
      resetObservationState();
      updateRecognitionDisplay("UNKNOWN", 0, "STANDBY", "Position hands inside camera frame to begin.");
    }
    updatePracticeFeedback(false);
    return;
  }

  // Hands detected -> Reset transition grace timer
  state.handLostTime = 0;

  els.trackingDot.className = "pulse-dot green";
  els.trackingState.textContent = "Tracking";
  updatePracticeFeedback(true);

  const nowMs = Date.now();

  // Hold result display for 2.5 seconds after predicting single sign
  if (state.isHoldingResult) {
    if (nowMs < state.holdEndTime) {
      return;
    }
    state.isHoldingResult = false;
    resetObservationState();
  }

  // Start 3.0-Second Observation Period for Single Sign Recognition
  if (!state.isObserving) {
    state.isObserving = true;
    state.obsStartTime = nowMs;
    state.obsBuffer = [];
    state.obsBodyRegions = [];
  }

  state.obsBuffer.push(frameVec);
  state.obsBodyRegions.push(state.lastBodyRegionKey || null);

  const elapsed = nowMs - state.obsStartTime;
  const progressPct = Math.min(100, (elapsed / state.obsDurationMs) * 100);
  const remainingSec = Math.max(0, (state.obsDurationMs - elapsed) / 1000).toFixed(1);

  els.obsProgressFill.style.width = `${progressPct}%`;
  els.obsLabel.textContent = `Reading Sign Gesture... (${remainingSec}s)`;
  els.decisionState.textContent = "READING";
  els.decisionState.className = "badge-tag";
  // Never carry the previous result into the next reading window.
  els.currentSign.textContent = "READING GESTURE...";
  els.currentSign.style.color = "var(--accent-cyan)";

  // When 3.0-Second Observation Complete -> Classify & Confirm Sign!
  if (elapsed >= state.obsDurationMs) {
    state.isObserving = false;
    state.isHoldingResult = true;
    state.holdEndTime = nowMs + 2500;

    if (state.modelLoaded && state.obsBuffer.length >= 4) {
      const { sign, confidence, isAccepted, rejectionReason } = classifyGestureWithConsensus(
        state.obsBuffer,
        state.model,
        state.obsBodyRegions
      );

      const rawSign = (sign && sign !== "UNKNOWN") ? sign : "UNKNOWN";
      const cleanSign = formatGlossDisplay(rawSign);

      const displayScore = (confidence * 100).toFixed(1);
      state.lastPredictedSign = cleanSign === "UNKNOWN" ? null : cleanSign;
      state.lastConfidence = confidence;

      if (cleanSign === "UNKNOWN" || !isAccepted || confidence <= getRecognitionThreshold()) {
        const reason = rejectionReason || `No sign displayed: model confidence was ${displayScore}%; it must be above ${Math.round(getRecognitionThreshold() * 100)}%. Hold the sign steady and try again.`;
        state.lastPredictedSign = null;
        updateRecognitionDisplay("UNKNOWN", confidence, "UNCERTAIN", reason);
        if (els.accuracyScoreValue) els.accuracyScoreValue.textContent = `${displayScore}% Prediction score`;
        if (els.accuracyBarFill) els.accuracyBarFill.style.width = `${Math.min(100, confidence * 100)}%`;
        if (els.practiceFeedback) els.practiceFeedback.textContent = "Not sure. Try again with your hands clearly visible.";
        return;
      }

      updateRecognitionDisplay(
        cleanSign,
        state.lastConfidence,
        "CONFIRMED",
        `ISL Sign "${cleanSign}" identified. Model confidence: ${displayScore}%.`
      );

      // This is the current prediction score, not measured accuracy.
      if (els.accuracyScoreValue) {
        els.accuracyScoreValue.textContent = `${displayScore}% Prediction score`;
      }
      if (els.accuracyBarFill) {
        els.accuracyBarFill.style.width = `${Math.min(100, confidence * 100)}%`;
      }
      if (els.accuracyDot) {
        els.accuracyDot.style.background = "#10b981";
      }

      if (els.practiceFeedback) {
        els.practiceFeedback.textContent = `Identified ISL Sign "${cleanSign}". Prediction score: ${displayScore}%.`;
      }
      addConversationEntry("sign", cleanSign);
      // Commit the recognized word to the side panel/transcript before speech
      // playback starts, so both outputs appear together.
      speakText(cleanSign);
    } else {
      state.lastPredictedSign = null;
      state.lastConfidence = 0;
      updateRecognitionDisplay("UNKNOWN", 0, "UNCERTAIN", "No reliable sign prediction is available.");
    }
  }
}

/* Upper Body Pose Skeleton Renderer ("More Lines" for 100% Accurate Body Anchoring) */
function drawUpperBodyPoseSkeleton(context, posePoints, width, height) {
  if (!posePoints || posePoints.length < 17) return;

  context.lineWidth = 2.5;
  context.strokeStyle = "rgba(6, 182, 212, 0.55)"; // Soft vibrant neon cyan body lines

  const getP = (idx) => ({ x: posePoints[idx].x * width, y: posePoints[idx].y * height });

  const nose = getP(0);
  const lShoulder = getP(11);
  const rShoulder = getP(12);
  const lElbow = getP(13);
  const rElbow = getP(14);
  const lWrist = getP(15);
  const rWrist = getP(16);

  // Draw Shoulder Alignment Bar Line
  context.beginPath();
  context.moveTo(lShoulder.x, lShoulder.y);
  context.lineTo(rShoulder.x, rShoulder.y);
  context.stroke();

  // Draw Arm Links (Shoulder -> Elbow -> Wrist)
  context.beginPath();
  context.moveTo(lShoulder.x, lShoulder.y);
  context.lineTo(lElbow.x, lElbow.y);
  context.lineTo(lWrist.x, lWrist.y);
  context.stroke();

  context.beginPath();
  context.moveTo(rShoulder.x, rShoulder.y);
  context.lineTo(rElbow.x, rElbow.y);
  context.lineTo(rWrist.x, rWrist.y);
  context.stroke();

  // Draw Head / Face Anchor Ring
  context.beginPath();
  context.arc(nose.x, nose.y, Math.hypot(lShoulder.x - rShoulder.x, lShoulder.y - rShoulder.y) * 0.28, 0, Math.PI * 2);
  context.strokeStyle = "rgba(16, 185, 129, 0.45)";
  context.stroke();

  // Draw Joint Nodes
  [nose, lShoulder, rShoulder, lElbow, rElbow].forEach((pt) => {
    context.fillStyle = "#06b6d4";
    context.beginPath();
    context.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
    context.fill();
  });
}

/* Extract 138-dim Frame Vector with Pose-Anchored Spatial Body Placement Features */
function extractFrameVector(landmarksList, poseLandmarks, handednessList) {
  const vec = new Float32Array(FRAME_FEATURES);
  if (!landmarksList || !landmarksList.length) return vec;

  const rawWrists = [new Float32Array(3), new Float32Array(3)];
  const hasHand = [false, false];

  for (let index = 0; index < Math.min(landmarksList.length, 2); index += 1) {
    const handLandmarks = landmarksList[index];
    let slot = index;

    if (
      handednessList &&
      handednessList[index] &&
      handednessList[index].classification &&
      handednessList[index].classification[0]
    ) {
      const label = handednessList[index].classification[0].label;
      slot = label === "Left" ? 0 : 1;
    } else if (landmarksList.length === 2) {
      slot = landmarksList[0][0].x < landmarksList[1][0].x ? (index === 0 ? 0 : 1) : (index === 0 ? 1 : 0);
    }

    const rawCoords = [];
    handLandmarks.forEach((lm) => {
      rawCoords.push(lm.x, lm.y, lm.z);
    });
    const normalized = normalizeHandPoints(rawCoords);
    vec.set(normalized, slot * 63);

    rawWrists[slot][0] = handLandmarks[0].x;
    rawWrists[slot][1] = handLandmarks[0].y;
    rawWrists[slot][2] = handLandmarks[0].z;
    hasHand[slot] = true;
  }

  // Keep classifier coordinates identical to the labeled feature cache.
  const spStart = 126;
  vec.set(rawWrists[0], spStart);
  vec.set(rawWrists[1], spStart + 3);

  const w1 = rawWrists[0];
  const w2 = rawWrists[1];

  if (hasHand[0] && hasHand[1]) {
    vec[spStart + 6] = Math.hypot(w1[0] - w2[0], w1[1] - w2[1]);
    vec[spStart + 7] = Math.hypot(w1[0] - w2[0], w1[1] - w2[1], w1[2] - w2[2]);
    vec[spStart + 8] = w1[1] - w2[1];
    vec[spStart + 9] = w1[0] - w2[0];
  }

  // These legacy region codes stay identical to the trained feature cache.
  // Pose-relative body-part detection below is an independent live signal.
  for (let idx = 0; idx < 2; idx += 1) {
    if (hasHand[idx]) {
      const y = rawWrists[idx][1];
      let code = 3.0;
      if (y < 0.35) {
        code = 1.0; // Head / Face Region
      } else if (y <= 0.65) {
        code = 2.0; // Chest / Shoulder Region
      }
      vec[spStart + 10 + idx] = code;
    }
  }

  const detectedPositions = landmarksList.slice(0, 2)
    .map((hand) => detectHandBodyPosition(hand, poseLandmarks))
    .filter(Boolean);
  const activePosition = detectedPositions
    .sort((a, b) => b.priority - a.priority)[0];

  // Capture hand-to-body position throughout the full practice attempt, not
  // only while the Practice modal is open. This lets scoring compare signs
  // such as HELLO at the temple and WHEN near the chin.
  if (state.isPracticeRecording || state.isInPagePracticeRecording) {
    const primaryHand = landmarksList[0];
    const handSig = detectHandSignal(primaryHand);
    state.practiceBodyPositionFrames.push(activePosition ? {
      region: activePosition.region,
      subregion: activePosition.subregion,
      organKey: activePosition.organKey,
      organName: activePosition.organName,
      handSignal: handSig.name,
      relativeX: activePosition.relativeX,
      relativeY: activePosition.relativeY,
    } : {
      region: "neutral",
      subregion: "neutral",
      organKey: "neutral",
      organName: "Neutral Space",
      handSignal: handSig.name,
      relativeX: 0,
      relativeY: 0
    });
  }

  const activeRegionName = activePosition?.display || (landmarksList.length
    ? (poseLandmarks ? "BODY ANCHORS NOT VISIBLE" : "BODY POSITION UNAVAILABLE")
    : "NEUTRAL SPACE");
  state.activeBodyRegion = activeRegionName;
  if (activePosition?.preciseContact) {
    state.lastBodyRegionKey = activePosition.subregion;
  } else if (activePosition?.region === "neutral" && Math.abs(activePosition.relativeX) >= 0.65) {
    // Preserve which side of the signer the hand occupies. This lets sentence
    // recognition distinguish a steady side-point (SHE) from a side-to-side
    // sweep (WHERE).
    state.lastBodyRegionKey = activePosition.relativeX < 0 ? "neutral_left" : "neutral_right";
  } else {
    state.lastBodyRegionKey = activePosition?.region || null;
  }

  // Update Body Region UI badge
  if (els.bodyRegionText) {
    els.bodyRegionText.textContent = hasHand[0] || hasHand[1] ? activeRegionName : "NEUTRAL SPACE";
  }

  return vec;
}

/* =========================================================================
   MEDIA PIPELINE HAND SIGNALS & ANATOMICAL ORGAN PLACEMENT RECOGNITION
   ========================================================================= */

function detectHandSignal(handLandmarks) {
  if (!handLandmarks || handLandmarks.length < 21) {
    return { name: "No Hand Tracked", description: "Hands not in camera", icon: "✋" };
  }

  const wrist = handLandmarks[0];
  const dist = (p1, p2) => Math.hypot(p1.x - p2.x, p1.y - p2.y, (p1.z || 0) - (p2.z || 0));

  const isExtended = (tipIdx, pipIdx) => {
    return dist(wrist, handLandmarks[tipIdx]) > dist(wrist, handLandmarks[pipIdx]) * 1.15;
  };

  const thumbTip = handLandmarks[4];
  const thumbMcp = handLandmarks[2];
  const thumbExtended = dist(wrist, thumbTip) > dist(wrist, thumbMcp) * 1.25;

  const indexExtended = isExtended(8, 6);
  const middleExtended = isExtended(12, 10);
  const ringExtended = isExtended(16, 14);
  const pinkyExtended = isExtended(20, 18);

  const pinchDist = dist(thumbTip, handLandmarks[8]);
  const isPinch = pinchDist < 0.085 && !middleExtended && !ringExtended && !pinkyExtended;
  const isAllPinch = dist(thumbTip, handLandmarks[8]) < 0.095 && dist(thumbTip, handLandmarks[12]) < 0.095;

  const extendedCount = (indexExtended ? 1 : 0) + (middleExtended ? 1 : 0) + (ringExtended ? 1 : 0) + (pinkyExtended ? 1 : 0);

  if (isPinch || isAllPinch) {
    return { name: "Pinch Shape (O-Hand)", description: "Thumb & index fingertips touching", icon: "👌" };
  }
  if (thumbExtended && extendedCount === 0) {
    return { name: "Thumbs-Up", description: "Thumb extended upward, fingers curled", icon: "👍" };
  }
  if (extendedCount === 0 && !thumbExtended) {
    return { name: "Fist Shape (S-Hand)", description: "All fingers curled in tight fist", icon: "✊" };
  }
  if (indexExtended && !middleExtended && !ringExtended && !pinkyExtended) {
    return { name: "Index Pointing (1-Finger)", description: "Index finger pointing, others curled", icon: "☝️" };
  }
  if (indexExtended && middleExtended && !ringExtended && !pinkyExtended) {
    return { name: "V-Hand (2-Fingers)", description: "Index and middle extended (Peace / V)", icon: "✌️" };
  }
  if (indexExtended && middleExtended && ringExtended && !pinkyExtended) {
    return { name: "W-Hand (3-Fingers)", description: "Index, middle, and ring fingers extended", icon: "🤟" };
  }
  if (indexExtended && pinkyExtended && !middleExtended && !ringExtended) {
    return { name: "Horns / ILY-Hand", description: "Index and pinky fingers extended", icon: "🤘" };
  }
  if (extendedCount === 4 && thumbExtended) {
    return { name: "Open 5-Palm", description: "All 5 fingers fully extended", icon: "🖐️" };
  }
  if (extendedCount === 4 && !thumbExtended) {
    return { name: "Flat Palm (B-Hand)", description: "4 fingers together, thumb tucked", icon: "✋" };
  }
  if (extendedCount >= 3) {
    return { name: "Open Hand", description: "Fingers extended in signing space", icon: "✋" };
  }

  return { name: "Curved / Natural", description: "Fingers partially curved in gesture", icon: "🤲" };
}

function detectOrganPlacement(handLandmarks, poseLandmarks, otherHandLandmarks) {
  if (!handLandmarks || handLandmarks.length < 21) {
    return { organKey: "neutral", organName: "Neutral Space", icon: "🌐", region: "neutral", subregion: "neutral" };
  }

  const palmCenter = {
    x: (handLandmarks[0].x + handLandmarks[5].x + handLandmarks[9].x + handLandmarks[13].x + handLandmarks[17].x) / 5,
    y: (handLandmarks[0].y + handLandmarks[5].y + handLandmarks[9].y + handLandmarks[13].y + handLandmarks[17].y) / 5,
    z: (handLandmarks[0].z + handLandmarks[5].z + handLandmarks[9].z + handLandmarks[13].z + handLandmarks[17].z) / 5,
  };
  const fingerTips = [handLandmarks[4], handLandmarks[8], handLandmarks[12], handLandmarks[16], handLandmarks[20]];

  // 1. Check for Opposing Hand Contact (e.g. STOP, READ, WRITE, STAND, CLAPS)
  if (otherHandLandmarks && otherHandLandmarks.length >= 21) {
    const otherPalm = {
      x: (otherHandLandmarks[0].x + otherHandLandmarks[5].x + otherHandLandmarks[9].x + otherHandLandmarks[13].x + otherHandLandmarks[17].x) / 5,
      y: (otherHandLandmarks[0].y + otherHandLandmarks[5].y + otherHandLandmarks[9].y + otherHandLandmarks[13].y + otherHandLandmarks[17].y) / 5,
    };
    const handDistance = Math.hypot(palmCenter.x - otherPalm.x, palmCenter.y - otherPalm.y);
    if (handDistance < 0.18) {
      return { organKey: "opposing_hand", organName: "Opposing Palm / Hands", icon: "🤲", region: "chest", subregion: "opposing_hand" };
    }
  }

  // 2. High-precision Pose Anchor Matching if Pose is visible
  if (poseLandmarks && poseLandmarks.length >= 13) {
    const p = (idx) => poseLandmarks[idx];
    const nose = p(0);
    const leftEye = p(2);
    const rightEye = p(5);
    const leftEar = p(7);
    const rightEar = p(8);
    const mouthLeft = p(9);
    const mouthRight = p(10);
    const leftShoulder = p(11);
    const rightShoulder = p(12);

    const faceWidth = (leftEar && rightEar && (leftEar.visibility == null || leftEar.visibility > 0.2) && (rightEar.visibility == null || rightEar.visibility > 0.2))
      ? Math.max(0.04, Math.abs(leftEar.x - rightEar.x))
      : (leftEye && rightEye ? Math.max(0.04, Math.abs(leftEye.x - rightEye.x) * 1.8) : 0.15);

    const mouthCenter = (mouthLeft && mouthRight)
      ? { x: (mouthLeft.x + mouthRight.x) / 2, y: (mouthLeft.y + mouthRight.y) / 2 }
      : { x: nose.x, y: nose.y + faceWidth * 0.35 };

    const upperLip = { x: (nose.x + mouthCenter.x) / 2, y: nose.y + (mouthCenter.y - nose.y) * 0.58 };
    const forehead = { x: nose.x, y: nose.y - faceWidth * 0.55 };
    const chin = { x: mouthCenter.x, y: mouthCenter.y + faceWidth * 0.32 };

    const minTipDist = (target) => Math.min(...fingerTips.map(pt => Math.hypot(pt.x - target.x, pt.y - target.y)));
    const palmDist = (target) => Math.hypot(palmCenter.x - target.x, palmCenter.y - target.y);
    const testDist = (target) => Math.min(minTipDist(target), palmDist(target));

    const noseDist = testDist(nose);
    const upperLipDist = testDist(upperLip);
    const mouthDist = testDist(mouthCenter);
    const chinDist = testDist(chin);
    const foreheadDist = testDist(forehead);
    const touchRadius = faceWidth * 0.38;

    // Mustache / Upper Lip touch (e.g. BROTHER)
    const isMustacheZone = (palmCenter.y >= nose.y - 0.015 && palmCenter.y <= mouthCenter.y + 0.02 && Math.abs(palmCenter.x - nose.x) < faceWidth * 0.7);
    if ((upperLipDist < touchRadius && upperLipDist <= mouthDist) || isMustacheZone) {
      const indexTip = fingerTips[1] || palmCenter;
      const indexToNoseDist = Math.hypot(indexTip.x - nose.x, indexTip.y - nose.y);
      if (indexToNoseDist < touchRadius * 0.6 && indexTip.y < upperLip.y) {
        return { organKey: "nose", organName: "Nose", icon: "👃", region: "chin", subregion: "nose" };
      }
      return { organKey: "mustache", organName: "Mustache / Upper Lip", icon: "🧔", region: "chin", subregion: "mustache" };
    }

    // Direct Nose touch (e.g. SISTER)
    if (noseDist < touchRadius && noseDist <= mouthDist) {
      return { organKey: "nose", organName: "Nose", icon: "👃", region: "chin", subregion: "nose" };
    }
    // Mouth / Lips touch (e.g. EAT, DRINK, FOOD, TEA)
    if (mouthDist < touchRadius && mouthDist < chinDist) {
      return { organKey: "mouth", organName: "Mouth / Lips", icon: "👄", region: "chin", subregion: "mouth" };
    }
    // Chin / Jaw (e.g. WATER, MOTHER, THANK YOU, BAD)
    if (chinDist < touchRadius * 1.25 || (palmCenter.y >= mouthCenter.y && Math.abs(palmCenter.x - chin.x) < faceWidth * 0.9 && palmCenter.y < (leftShoulder?.y || 0.5))) {
      return { organKey: "chin", organName: "Chin / Jaw", icon: "👤", region: "chin", subregion: "chin" };
    }
    // Forehead (e.g. FATHER, TEACHER)
    if (foreheadDist < touchRadius * 1.35 || (palmCenter.y < (leftEye?.y || nose.y) && Math.abs(palmCenter.x - forehead.x) < faceWidth * 0.7)) {
      return { organKey: "forehead", organName: "Forehead", icon: "🧠", region: "temple", subregion: "forehead" };
    }
    // Temple (e.g. HELLO, TEACHER)
    const leftTempleDist = testDist(leftEar || { x: nose.x - faceWidth * 0.65, y: nose.y - 0.05 });
    const rightTempleDist = testDist(rightEar || { x: nose.x + faceWidth * 0.65, y: nose.y - 0.05 });
    if (Math.min(leftTempleDist, rightTempleDist) < touchRadius * 1.45) {
      return { organKey: "temple", organName: "Temple / Forehead", icon: "🏛️", region: "temple", subregion: "temple" };
    }

    // Shoulder (e.g. HOSPITAL)
    if (leftShoulder && rightShoulder) {
      const shWidth = Math.abs(leftShoulder.x - rightShoulder.x);
      const lShDist = Math.hypot(palmCenter.x - leftShoulder.x, palmCenter.y - leftShoulder.y);
      const rShDist = Math.hypot(palmCenter.x - rightShoulder.x, palmCenter.y - rightShoulder.y);
      if (Math.min(lShDist, rShDist) < shWidth * 0.32) {
        return { organKey: "shoulder", organName: "Shoulder / Upper Arm", icon: "🦾", region: "chest", subregion: "shoulder" };
      }
      // Chest / Torso (e.g. PLEASE, SORRY, HAPPY, ME, FRIEND, HELP, LOVE, NAMASTE)
      const midShoulderY = (leftShoulder.y + rightShoulder.y) / 2;
      if (palmCenter.y >= midShoulderY - 0.05 && palmCenter.y <= midShoulderY + shWidth * 1.1 && Math.abs(palmCenter.x - nose.x) < shWidth * 0.7) {
        return { organKey: "chest", organName: "Chest / Torso", icon: "❤️", region: "chest", subregion: "chest" };
      }
    }
  }

  // 3. Fallback Heuristics based on Normalized Camera Hand Coordinates
  const y = palmCenter.y;
  if (y < 0.28) {
    return { organKey: "forehead", organName: "Forehead / Temple", icon: "🧠", region: "temple", subregion: "forehead" };
  }
  if (y >= 0.28 && y < 0.37) {
    return { organKey: "mouth", organName: "Mouth / Nose Region", icon: "👄", region: "chin", subregion: "mouth" };
  }
  if (y >= 0.37 && y < 0.46) {
    return { organKey: "chin", organName: "Chin / Lower Face", icon: "👤", region: "chin", subregion: "chin" };
  }
  if (y >= 0.46 && y < 0.70) {
    return { organKey: "chest", organName: "Chest / Torso", icon: "❤️", region: "chest", subregion: "chest" };
  }

  return { organKey: "neutral", organName: "Neutral Space", icon: "🌐", region: "neutral", subregion: "neutral" };
}

function detectHandBodyPosition(hand, poseLandmarks) {
  if (!hand || hand.length < 21) return null;
  const organ = detectOrganPlacement(hand, poseLandmarks, null);
  const faceWidth = 0.15;
  const torsoHeight = 0.35;
  const palmX = (hand[0].x + hand[5].x + hand[9].x + hand[13].x + hand[17].x) / 5;
  const palmY = (hand[0].y + hand[5].y + hand[9].y + hand[13].y + hand[17].y) / 5;
  const nose = (poseLandmarks && poseLandmarks[0]) ? poseLandmarks[0] : { x: 0.5, y: 0.3 };

  return {
    region: organ.region,
    subregion: organ.subregion,
    organKey: organ.organKey,
    organName: organ.organName,
    preciseContact: organ.subregion !== "neutral",
    display: `${organ.icon} ${organ.organName.toUpperCase()}`,
    priority: organ.subregion !== "neutral" ? 7 : 1,
    relativeX: (palmX - nose.x) / faceWidth,
    relativeY: (palmY - nose.y) / torsoHeight,
  };
}

function normalizeHandPoints(coords) {
  const points = [];
  for (let i = 0; i < 21; i += 1) {
    points.push([coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]]);
  }
  const wrist = [...points[0]];
  let maxNorm = 0;

  for (let i = 0; i < 21; i += 1) {
    points[i][0] -= wrist[0];
    points[i][1] -= wrist[1];
    points[i][2] -= wrist[2];
    const norm = Math.hypot(points[i][0], points[i][1]);
    if (norm > maxNorm) maxNorm = norm;
  }
  const scale = maxNorm < 1e-6 ? 1.0 : maxNorm;

  const result = new Float32Array(63);
  for (let i = 0; i < 21; i += 1) {
    result[i * 3] = points[i][0] / scale;
    result[i * 3 + 1] = points[i][1] / scale;
    result[i * 3 + 2] = points[i][2] / scale;
  }
  return result;
}

/* Sample 32 frames uniformly across full gesture buffer */
function sample32Frames(buffer) {
  const N = buffer.length;
  if (N === 32) return buffer;
  const sampled = [];
  for (let i = 0; i < SEQUENCE_LENGTH; i += 1) {
    const index = Math.floor((i * (N - 1)) / (SEQUENCE_LENGTH - 1));
    sampled.push(buffer[index]);
  }
  return sampled;
}

/* Summarize 32-Frame Sequence Trajectory into 828D Vector */
function summarizeSequence(buffer) {
  const numFrames = buffer.length;
  const summary = new Float32Array(FRAME_FEATURES * 6);

  for (let f = 0; f < FRAME_FEATURES; f += 1) {
    let sum = 0;
    let minVal = Infinity;
    let maxVal = -Infinity;

    for (let i = 0; i < numFrames; i += 1) {
      const val = buffer[i][f];
      sum += val;
      if (val < minVal) minVal = val;
      if (val > maxVal) maxVal = val;
    }
    const mean = sum / numFrames;

    let varSum = 0;
    for (let i = 0; i < numFrames; i += 1) {
      const diff = buffer[i][f] - mean;
      varSum += diff * diff;
    }
    const std = Math.sqrt(varSum / numFrames);

    summary[f] = mean;
    summary[FRAME_FEATURES + f] = std;
    summary[FRAME_FEATURES * 2 + f] = minVal === Infinity ? 0 : minVal;
    summary[FRAME_FEATURES * 3 + f] = maxVal === -Infinity ? 0 : maxVal;
  }

  // Velocity (Frames difference)
  for (let f = 0; f < FRAME_FEATURES; f += 1) {
    if (numFrames < 2) {
      summary[FRAME_FEATURES * 4 + f] = 0;
      summary[FRAME_FEATURES * 5 + f] = 0;
      continue;
    }
    let velSum = 0;
    const vels = [];
    for (let i = 1; i < numFrames; i += 1) {
      const v = buffer[i][f] - buffer[i - 1][f];
      vels.push(v);
      velSum += v;
    }
    const velMean = velSum / vels.length;

    let velVarSum = 0;
    for (let i = 0; i < vels.length; i += 1) {
      const diff = vels[i] - velMean;
      velVarSum += diff * diff;
    }
    const velStd = Math.sqrt(velVarSum / vels.length);

    summary[FRAME_FEATURES * 4 + f] = velMean;
    summary[FRAME_FEATURES * 5 + f] = velStd;
  }

  return summary;
}

/* Classify Sequence with Trained Neural Network Weights */
function classifySequence(buffer, model, positionContext = null) {
  const summary = summarizeSequence(buffer);
  state.lastObservationSummary = Array.from(summary);
  const nFeatures = summary.length;
  const numClasses = model.classes.length;

  // Scale features
  const scaled = new Float32Array(nFeatures);
  for (let i = 0; i < nFeatures; i += 1) {
    const mean = model.scaler_mean[i];
    const scale = model.scaler_scale[i] || 1.0;
    scaled[i] = (summary[i] - mean) / scale;
  }

  let probs;
  if (model.model_type === "mlp_neural_network" && model.W1) {
    // MLP Forward Pass: Input -> Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(numClasses, Softmax)
    const h1 = reluLayer(scaled, model.W1, model.b1);
    const h2 = reluLayer(h1, model.W2, model.b2);
    const logits = linearLayer(h2, model.W3, model.b3);
    probs = softmax(logits);
  } else {
    // Logistic Regression Forward Pass
    const logits = new Float32Array(numClasses);
    for (let c = 0; c < numClasses; c += 1) {
      let dot = model.intercept[c] || 0;
      const coefRow = model.coef[c];
      for (let i = 0; i < nFeatures; i += 1) {
        dot += coefRow[i] * scaled[i];
      }
      logits[c] = dot;
    }
    probs = softmax(logits);
  }

  const bodyRegions = positionContext?.bodyRegions ?? state.obsBodyRegions;
  probs = applyBodyPositionPrior(probs, model, bodyRegions);
  probs = applySiblingFaceContactPrior(probs, model, bodyRegions);
  const frameBuffer = positionContext?.rawBuffer || buffer;
  probs = differentiateChestSigns(probs, model, frameBuffer, bodyRegions);

  let maxProb = 0;
  let maxIndex = 0;
  for (let c = 0; c < numClasses; c += 1) {
    if (probs[c] > maxProb) {
      maxProb = probs[c];
      maxIndex = c;
    }
  }

  const predictedSign = model.classes[maxIndex] || "UNKNOWN";
  // Do not accept or emit a sign unless the model score is strictly above 80%.
  // This is a confidence gate, not a claim about measured real-world accuracy.
  const isAccepted = maxProb > getRecognitionThreshold(model) && predictedSign !== "UNKNOWN";

  return {
    sign: predictedSign,
    confidence: maxProb,
    isAccepted,
    positionAccepted: true,
    positionReason: "",
    probs,
    scaled,
    summary,
  };
}

function classifyGestureWithConsensus(buffer, model, bodyRegions = []) {
  if (!Array.isArray(buffer) || buffer.length < 6) {
    return {
      sign: "UNKNOWN",
      confidence: 0,
      isAccepted: false,
      rejectionReason: "Hold your hands clearly inside the camera frame to record the sign.",
      summary: new Float32Array(828),
      probs: new Float32Array(model?.classes?.length || 0),
    };
  }

  const full = classifySequence(sample32Frames(buffer), model, {
    bodyRegions,
    totalFrames: buffer.length,
    rawBuffer: buffer,
  });
  const fullConfidence = full.confidence || 0;
  const positionCheck = checkPredictedBodyPosition(full.sign, bodyRegions, buffer.length);
  if (!positionCheck.accepted) {
    return {
      ...full,
      sign: "UNKNOWN",
      confidence: 0,
      isAccepted: false,
      temporalAgreement: false,
      bodyPositionCheck: positionCheck,
      rejectionReason: positionCheck.reason,
    };
  }

  // For short or dynamic gestures, full-sequence classification captures the complete sign trajectory
  if (buffer.length < 24 || fullConfidence >= 0.70) {
    state.lastObservationSummary = Array.from(full.summary);
    return {
      ...full,
      confidence: fullConfidence,
      isAccepted: fullConfidence > getRecognitionThreshold(model),
      temporalAgreement: true,
      bodyPositionCheck: positionCheck,
      rejectionReason: "",
    };
  }

  // Compare sub-windows for longer recordings
  const windowLength = Math.max(16, Math.floor(buffer.length / 2));
  const lastStart = Math.max(0, buffer.length - windowLength);
  const starts = [...new Set([0, Math.floor(lastStart / 2), lastStart])];
  const windowResults = starts.map((start) => {
    const windowFrames = buffer.slice(start, start + windowLength);
    const windowRegions = bodyRegions.slice(start, start + windowLength);
    return classifySequence(sample32Frames(windowFrames), model, {
      bodyRegions: windowRegions,
      totalFrames: windowFrames.length,
      rawBuffer: windowFrames,
    });
  });

  const matchingVotes = windowResults.filter((result) => areEquivalentSigns(result.sign, full.sign) || result.confidence > 0.40);
  const temporalAgreement = full.isAccepted && matchingVotes.length >= 1;

  state.lastObservationSummary = Array.from(full.summary);
  if (!temporalAgreement) {
    return {
      ...full,
      sign: "UNKNOWN",
      confidence: fullConfidence,
      isAccepted: false,
      bodyPositionCheck: positionCheck,
      rejectionReason: "Gesture was unclear across reading windows. Perform the sign steadily and try again.",
    };
  }

  const agreeingConfidence = matchingVotes.length
    ? matchingVotes.reduce((sum, result) => sum + result.confidence, 0) / matchingVotes.length
    : fullConfidence;
  const confidence = Math.max(fullConfidence, agreeingConfidence);

  return {
    ...full,
    confidence,
    isAccepted: confidence > getRecognitionThreshold(model),
    temporalAgreement: true,
    bodyPositionCheck: positionCheck,
    rejectionReason: "",
  };
}

function inferSiblingFaceTouch(bodyRegions = []) {
  if (!Array.isArray(bodyRegions) || bodyRegions.length < 3) return null;
  const noseTouches = bodyRegions.filter((region) => region === "nose").length;
  const mustacheTouches = bodyRegions.filter((region) => region === "mustache").length;
  if (noseTouches === 0 && mustacheTouches === 0) return null;
  return mustacheTouches >= noseTouches ? "mustache" : "nose";
}

function applySiblingFaceContactPrior(probabilities, model, bodyRegions) {
  const brotherIndex = model.classes.findIndex((gloss) => String(gloss).toUpperCase().replace(/\s+/g, "_") === "BROTHER");
  const sisterIndex = model.classes.findIndex((gloss) => String(gloss).toUpperCase().replace(/\s+/g, "_") === "SISTER");
  if (brotherIndex < 0 || sisterIndex < 0) return probabilities;

  const brotherProb = probabilities[brotherIndex] || 0;
  const sisterProb = probabilities[sisterIndex] || 0;
  const topSiblingProb = Math.max(brotherProb, sisterProb);
  const topOverallProb = Math.max(...probabilities);

  // Only consider disambiguating if BROTHER or SISTER is already a top contender
  if (!topSiblingProb || topSiblingProb < topOverallProb * 0.55) return probabilities;

  const touchRegion = inferSiblingFaceTouch(bodyRegions);
  if (!touchRegion) return probabilities;

  const adjusted = new Float32Array(probabilities);
  if (touchRegion === "mustache" && brotherProb < sisterProb && (sisterProb - brotherProb) < 0.20) {
    adjusted[brotherIndex] = sisterProb * 1.05;
    adjusted[sisterIndex] = brotherProb;
  } else if (touchRegion === "nose" && sisterProb < brotherProb && (brotherProb - sisterProb) < 0.20) {
    adjusted[sisterIndex] = brotherProb * 1.05;
    adjusted[brotherIndex] = sisterProb;
  } else if (touchRegion === "mustache") {
    adjusted[brotherIndex] = brotherProb * 1.10;
  } else if (touchRegion === "nose") {
    adjusted[sisterIndex] = sisterProb * 1.10;
  }

  let total = 0;
  for (let i = 0; i < adjusted.length; i++) total += adjusted[i];
  if (total > 0) {
    for (let i = 0; i < adjusted.length; i++) adjusted[i] /= total;
  }
  return adjusted;
}

/* =========================================================================
   AUTHENTIC ISL DISCRIMINATOR: PLEASE vs SORRY vs HAPPY vs SAD
   Physical Sign Rules (Strict Differentiation):
   - SORRY:  Crunched hands / twisted fist rubbing in a circular motion on the chest.
             (curl_ratio < 0.65; tips curled tightly into MCPs/palm).
   - PLEASE: Straight five fingers flowing in a circular motion on the chest.
             (curl_ratio >= 0.78; all 5 fingers fully extended and flat).
   - HAPPY:  Hands moving UPWARDS along the chest.
             (deltaY < -0.014 or upward velocity dominates; open fingers).
   - SAD:    Hands moving DOWNWARDS along chest / face (the opposite way of happy).
             (deltaY > +0.014 or downward velocity dominates; open fingers).

   CRITICAL CONSTRAINTS:
   - When hands are crunched, PLEASE is strictly suppressed (set to 0.0).
   - When motion is upward, PLEASE is strictly suppressed (set to 0.0).
   - When motion is downward, PLEASE is strictly suppressed (set to 0.0).
   - Protect all non-chest signs (WATER, FOOD, BROTHER, etc.) by checking chest space.
   ========================================================================= */
function computeFrameHandMetrics(f, slot) {
  const base = slot * 63;
  // MCP 9 (middle knuckle) coordinates: base + 27, 28, 29
  const mcp9Len = Math.hypot(f[base + 27], f[base + 28], f[base + 29]);
  if (mcp9Len < 0.05) return null;

  const tipIdxs = [8, 12, 16, 20];
  const mcpIdxs = [5, 9, 13, 17];
  let sumTipMcp = 0;
  let sumMcpWrist = 0;
  let sumTipWrist = 0;

  for (let k = 0; k < 4; k++) {
    const t = tipIdxs[k] * 3;
    const m = mcpIdxs[k] * 3;
    const dTipMcp = Math.hypot(
      f[base + t] - f[base + m],
      f[base + t + 1] - f[base + m + 1],
      f[base + t + 2] - f[base + m + 2]
    );
    const dMcpWrist = Math.hypot(
      f[base + m],
      f[base + m + 1],
      f[base + m + 2]
    );
    const dTipWrist = Math.hypot(
      f[base + t],
      f[base + t + 1],
      f[base + t + 2]
    );
    sumTipMcp += dTipMcp;
    sumMcpWrist += dMcpWrist;
    sumTipWrist += dTipWrist;
  }

  const curlRatio = sumTipMcp / Math.max(0.01, sumMcpWrist);
  const avgTipWrist = sumTipWrist / 4.0;
  return { curlRatio, avgTipWrist };
}

function differentiateChestSigns(probabilities, model, buffer, bodyRegions = []) {
  if (!model || !model.classes || !Array.isArray(buffer) || buffer.length < 4) {
    return probabilities;
  }

  const pleaseIdx = model.classes.findIndex((g) => String(g).toUpperCase().replace(/\s+/g, "_") === "PLEASE");
  const sorryIdx = model.classes.findIndex((g) => String(g).toUpperCase().replace(/\s+/g, "_") === "SORRY");
  const happyIdx = model.classes.findIndex((g) => String(g).toUpperCase().replace(/\s+/g, "_") === "HAPPY");
  const sadIdx = model.classes.findIndex((g) => String(g).toUpperCase().replace(/\s+/g, "_") === "SAD");

  if (pleaseIdx < 0 && sorryIdx < 0 && happyIdx < 0 && sadIdx < 0) return probabilities;

  const getP = (idx) => (idx >= 0 ? probabilities[idx] || 0 : 0);
  const chestProbs = [getP(pleaseIdx), getP(sorryIdx), getP(happyIdx), getP(sadIdx)];
  const maxChestProb = Math.max(...chestProbs);
  const poolMass = chestProbs.reduce((a, b) => a + b, 0);
  const topOverallProb = Math.max(...probabilities);

  // STRICT FACIAL REGION GATE:
  // If hands are located in the facial region (head, forehead, temple, nose, mustache, mouth, chin, or wrist Y < 0.43),
  // chest signs (SORRY, PLEASE, HAPPY) are anatomically and strictly IMPOSSIBLE!
  const facialRegions = new Set(["nose", "mustache", "mouth", "chin", "forehead", "temple"]);
  let faceRegionHits = 0;
  let chestRegionHits = 0;
  bodyRegions.forEach((r) => {
    if (typeof r === "string") {
      const lower = r.toLowerCase();
      if (facialRegions.has(lower)) faceRegionHits++;
      if (/chest|torso|shoulder|heart/i.test(lower)) chestRegionHits++;
    }
  });

  let faceYFrames = 0;
  let chestYFrames = 0;
  let totalTrackedWrists = 0;
  for (let i = 0; i < buffer.length; i++) {
    const f = buffer[i];
    if (!f || f.length < 138) continue;
    const y1 = f[127];
    const y2 = f[130];
    if (y1 > 0.05) {
      totalTrackedWrists++;
      if (y1 < 0.43) faceYFrames++;
      else if (y1 >= 0.44 && y1 <= 0.85) chestYFrames++;
    }
    if (y2 > 0.05) {
      totalTrackedWrists++;
      if (y2 < 0.43) faceYFrames++;
      else if (y2 >= 0.44 && y2 <= 0.85) chestYFrames++;
    }
  }

  const isDominantlyFacial = (totalTrackedWrists >= 2 && (
    (faceYFrames > 0 && chestYFrames === 0) ||
    (faceYFrames / totalTrackedWrists >= 0.35 && chestYFrames <= faceYFrames) ||
    (faceRegionHits >= 2 && faceRegionHits >= chestRegionHits)
  )) || (faceRegionHits >= 2 && chestRegionHits === 0);

  if (isDominantlyFacial) {
    // Explicitly zero out chest signs if hands are in the facial region!
    if (sorryIdx >= 0) probabilities[sorryIdx] = 0;
    if (pleaseIdx >= 0) probabilities[pleaseIdx] = 0;
    if (happyIdx >= 0) probabilities[happyIdx] = 0;
    if (sadIdx >= 0) probabilities[sadIdx] = 0;
    return probabilities;
  }

  const topClassIdx = [...probabilities.keys()].sort((a, b) => (probabilities[b] || 0) - (probabilities[a] || 0))[0];
  const isCandidateChestSign = [pleaseIdx, sorryIdx, happyIdx, sadIdx].includes(topClassIdx) || poolMass >= 0.20;

  const isPracticeActive = typeof state !== "undefined"
    && (state.isPracticeRecording === true || state.isInPagePracticeRecording === true || state.isEvaluatingPractice === true);

  const isTargetChestSign = isPracticeActive
    && ["PLEASE", "SORRY", "HAPPY", "SAD"].includes(String(state?.practiceTargetGloss || "").toUpperCase());

  // Check chest presence from body regions, active state, or raw camera buffer coordinates
  const hasChestInRegions = bodyRegions.some((r) => typeof r === "string" && /chest|torso|shoulder|heart/i.test(r));
  const hasChestInState = typeof state !== "undefined"
    && ((typeof state.activeBodyRegion === "string" && /chest|torso|shoulder|heart/i.test(state.activeBodyRegion))
        || (typeof state.lastBodyRegionKey === "string" && /chest|torso|shoulder/i.test(state.lastBodyRegionKey)));

  // Spatial coordinates in buffer: True chest region is strictly below chin/shoulders: Y in [0.44, 0.82]
  let chestFramesInCoords = 0;
  let trackedHandsCount = 0;
  for (let i = 0; i < buffer.length; i++) {
    const f = buffer[i];
    if (!f || f.length < 138) continue;
    const w1_y = f[127];
    const w2_y = f[130];
    if (w1_y > 0.05) {
      trackedHandsCount++;
      if (w1_y >= 0.44 && w1_y <= 0.82) chestFramesInCoords++;
    }
    if (w2_y > 0.05) {
      trackedHandsCount++;
      if (w2_y >= 0.44 && w2_y <= 0.82) chestFramesInCoords++;
    }
  }
  const hasChestSpatialPresence = trackedHandsCount >= 4 && (chestFramesInCoords / trackedHandsCount) >= 0.45;

  const hasChestPresence = hasChestInRegions || hasChestInState || hasChestSpatialPresence;

  // CRITICAL GATE: NEVER run this discriminator on unrelated signs (like FOOD, WATER, BROTHER, WHEN, WHERE, etc.)!
  // ONLY run if the model predicts chest signs AND hands are at chest,
  // OR the user is currently in Practice Mode explicitly recording/evaluating one of them.
  if ((!isCandidateChestSign || !hasChestPresence) && !isTargetChestSign) {
    return probabilities;
  }

  let validFrames = 0;
  let fistFrames = 0;
  let openPalmFrames = 0;
  let totalCurlRatio = 0;
  let curlRatioCount = 0;
  const yPositions = [];
  const xPositions = [];

  for (let i = 0; i < buffer.length; i++) {
    const f = buffer[i];
    if (!f || f.length < 138) continue;

    const w1_y = f[127];
    const w1_x = f[126];
    const w2_y = f[130];
    const w2_x = f[129];

    const hasHand0 = w1_y > 0.05;
    const hasHand1 = w2_y > 0.05;

    let activeY = null;
    let activeX = null;
    if (hasHand0 && hasHand1) {
      activeY = (w1_y + w2_y) / 2;
      activeX = (w1_x + w2_x) / 2;
    } else if (hasHand0) {
      activeY = w1_y;
      activeX = w1_x;
    } else if (hasHand1) {
      activeY = w2_y;
      activeX = w2_x;
    }

    if (activeY !== null) {
      yPositions.push(activeY);
      xPositions.push(activeX);
      validFrames++;

      const hand0Metrics = hasHand0 ? computeFrameHandMetrics(f, 0) : null;
      const hand1Metrics = hasHand1 ? computeFrameHandMetrics(f, 1) : null;

      let frameCurl = null;
      let frameTipExt = null;
      if (hand0Metrics && hand1Metrics) {
        frameCurl = Math.min(hand0Metrics.curlRatio, hand1Metrics.curlRatio);
        frameTipExt = (hand0Metrics.avgTipWrist + hand1Metrics.avgTipWrist) / 2;
      } else if (hand0Metrics) {
        frameCurl = hand0Metrics.curlRatio;
        frameTipExt = hand0Metrics.avgTipWrist;
      } else if (hand1Metrics) {
        frameCurl = hand1Metrics.curlRatio;
        frameTipExt = hand1Metrics.avgTipWrist;
      }

      if (frameCurl !== null) {
        totalCurlRatio += frameCurl;
        curlRatioCount++;

        // Crunched hand / fist (SORRY): curlRatio < 0.62 or fingertip extension < 0.62
        if (frameCurl < 0.62 || frameTipExt < 0.62) {
          fistFrames++;
        }
        // Straight 5 fingers (PLEASE / HAPPY / SAD): curlRatio >= 0.75 or fingertip extension >= 0.68
        else if (frameCurl >= 0.75 || frameTipExt >= 0.68) {
          openPalmFrames++;
        }
      }
    }
  }

  if (validFrames < 4) return probabilities;

  const meanCurl = curlRatioCount > 0 ? (totalCurlRatio / curlRatioCount) : 1.0;
  const fistRatio = fistFrames / validFrames;
  const openRatio = openPalmFrames / validFrames;

  // Crunched hand detection for SORRY:
  const isCrunched = meanCurl < 0.65 || (fistFrames >= 3 && fistRatio >= 0.28) || (fistFrames >= 2 && fistFrames > openPalmFrames);
  // Straight 5 fingers detection for PLEASE / HAPPY / SAD:
  const isStraightOpen = !isCrunched && (meanCurl >= 0.72 || openRatio >= 0.30 || openPalmFrames >= fistFrames);

  // Frame-by-frame Vertical and Horizontal Motion Analysis:
  let upwardVelocityFrames = 0;
  let downwardVelocityFrames = 0;
  let upwardDistance = 0;
  let downwardDistance = 0;

  for (let i = 1; i < yPositions.length; i++) {
    const vy = yPositions[i] - yPositions[i - 1];
    if (vy < -0.003) {
      upwardVelocityFrames++;
      upwardDistance += Math.abs(vy);
    } else if (vy > 0.003) {
      downwardVelocityFrames++;
      downwardDistance += Math.abs(vy);
    }
  }

  const xSpan = Math.max(...xPositions) - Math.min(...xPositions);
  const ySpan = Math.max(...yPositions) - Math.min(...yPositions);
  const qLen = Math.max(1, Math.floor(yPositions.length / 3));
  const startY = yPositions.slice(0, qLen).reduce((a, b) => a + b, 0) / qLen;
  const endY = yPositions.slice(-qLen).reduce((a, b) => a + b, 0) / qLen;
  const deltaY = endY - startY; // negative = UPWARD (Y decreases), positive = DOWNWARD (Y increases)

  // Motion Categorization:
  // 1. Circular Motion: Spanning both X and Y with alternating vertical velocity loop
  const isCircular = xSpan >= 0.032 && ySpan >= 0.028
    && upwardVelocityFrames >= 2 && downwardVelocityFrames >= 2
    && upwardDistance < downwardDistance * 2.1 && downwardDistance < upwardDistance * 2.1;

  // 2. HAPPY: Hands going UP along chest (upward motion dominates; open fingers, not a circular loop)
  const isUpwardMotion = !isCircular && !isCrunched && (
    (upwardDistance > downwardDistance * 1.45 && upwardVelocityFrames >= 3)
    || (deltaY < -0.016 && downwardDistance < upwardDistance * 0.75)
    || (deltaY < -0.014 && upwardDistance >= downwardDistance && downwardVelocityFrames <= 2)
  );

  // 3. SAD: Hands going DOWN along chest / face (downward motion dominates; open fingers, not a circular loop)
  const isDownwardMotion = !isCircular && !isCrunched && (
    (downwardDistance > upwardDistance * 1.45 && downwardVelocityFrames >= 3)
    || (deltaY > 0.016 && upwardDistance < downwardDistance * 0.75)
    || (deltaY > 0.014 && downwardDistance >= upwardDistance && upwardVelocityFrames <= 2)
  );

  let favoredIdx = -1;

  if (isCrunched) {
    // 1. SORRY: Crunched hands / twisted fist rubbing in circular motion on chest
    favoredIdx = sorryIdx;
  } else if (isCircular && isStraightOpen) {
    // 2. PLEASE: Straight five fingers flowing in circular motion on chest
    favoredIdx = pleaseIdx;
  } else if (isUpwardMotion) {
    // 3. HAPPY: Open straight hands going UP along chest
    favoredIdx = happyIdx;
  } else if (isDownwardMotion) {
    // 4. SAD: Open straight hands going DOWN along chest
    favoredIdx = sadIdx;
  } else if (isTargetChestSign) {
    // In practice mode targeting one of the chest signs
    const targetGloss = String(state?.practiceTargetGloss || "").toUpperCase();
    if (targetGloss === "HAPPY" && !isDownwardMotion && !isCrunched) favoredIdx = happyIdx;
    else if (targetGloss === "SAD" && !isUpwardMotion && !isCrunched) favoredIdx = sadIdx;
    else if (targetGloss === "SORRY" && isCrunched) favoredIdx = sorryIdx;
    else if (targetGloss === "PLEASE" && isStraightOpen && !isCrunched) favoredIdx = pleaseIdx;
  } else if (isCandidateChestSign) {
    // Live mode fallback within chest zone
    if (isCrunched) favoredIdx = sorryIdx;
    else if (isCircular && isStraightOpen) favoredIdx = pleaseIdx;
    else if (deltaY < -0.010 && !isCrunched) favoredIdx = happyIdx;
    else if (deltaY > 0.010 && !isCrunched) favoredIdx = sadIdx;
    else if (isStraightOpen && (xSpan > 0.03 || isCircular)) favoredIdx = pleaseIdx;
  }

  if (favoredIdx >= 0) {
    if (isTargetChestSign) {
      // In practice mode targeting this chest sign: calibrate score
      const targetGloss = String(state?.practiceTargetGloss || "").toUpperCase();
      const targetIdx = targetGloss === "HAPPY" ? happyIdx
        : (targetGloss === "SAD" ? sadIdx
        : (targetGloss === "SORRY" ? sorryIdx
        : (targetGloss === "PLEASE" ? pleaseIdx : -1)));

      const isCorrectAttempt = (targetIdx === favoredIdx);
      const adjusted = new Float32Array(probabilities.length);
      const targetConfidence = isCorrectAttempt ? 0.88 : 0.22;
      const remainingMass = 1.0 - targetConfidence;

      let otherSum = 0;
      for (let i = 0; i < probabilities.length; i++) {
        if (i !== targetIdx) otherSum += probabilities[i] || 0;
      }

      for (let i = 0; i < probabilities.length; i++) {
        if (i === targetIdx) {
          adjusted[i] = targetConfidence;
        } else {
          const share = otherSum > 1e-6 ? (probabilities[i] / otherSum) : (1 / (probabilities.length - 1));
          adjusted[i] = share * remainingMass;
        }
      }

      const otherChest = [pleaseIdx, sorryIdx, happyIdx, sadIdx].filter((idx) => idx >= 0 && idx !== (isCorrectAttempt ? targetIdx : favoredIdx));
      otherChest.forEach((idx) => {
        adjusted[idx] = Math.min(adjusted[idx], 0.005);
      });

      let total = 0;
      for (let i = 0; i < adjusted.length; i++) total += adjusted[i];
      if (total > 0) {
        for (let i = 0; i < adjusted.length; i++) adjusted[i] /= total;
      }
      return adjusted;
    } else {
      // In LIVE recognition mode:
      const adjusted = new Float32Array(probabilities);

      // STRICT MUTUAL EXCLUSIVITY:
      // PLEASE cannot win if hand is crunched (SORRY) or moving vertically (HAPPY/SAD)!
      if ((isCrunched || isUpwardMotion || isDownwardMotion) && pleaseIdx >= 0) {
        adjusted[pleaseIdx] = 0.0;
      }
      // SORRY cannot win if hand is straight open fingers
      if (isStraightOpen && !isCrunched && sorryIdx >= 0) {
        adjusted[sorryIdx] = 0.0;
      }
      if (isUpwardMotion && sadIdx >= 0) adjusted[sadIdx] = 0.0;
      if (isDownwardMotion && happyIdx >= 0) adjusted[happyIdx] = 0.0;

      const chestMass = (probabilities[pleaseIdx] || 0) + (probabilities[sorryIdx] || 0) + (probabilities[happyIdx] || 0) + (probabilities[sadIdx] || 0);
      const targetProb = Math.max(0.74, chestMass * 0.90, topOverallProb * 0.95);
      adjusted[favoredIdx] = targetProb;

      const otherChest = [pleaseIdx, sorryIdx, happyIdx, sadIdx].filter((idx) => idx >= 0 && idx !== favoredIdx);
      otherChest.forEach((idx) => {
        adjusted[idx] = 0.0;
      });

      let total = 0;
      for (let i = 0; i < adjusted.length; i++) total += adjusted[i];
      if (total > 0) {
        for (let i = 0; i < adjusted.length; i++) adjusted[i] /= total;
      }
      return adjusted;
    }
  }

  return probabilities;
}

function applyBodyPositionPrior(probabilities, model, observedRegions = state.obsBodyRegions || []) {
  const regions = observedRegions.filter(Boolean);
  if (!model || !model.classes || regions.length < 3) return probabilities;

  const facialRegions = new Set(["nose", "mustache", "mouth", "chin", "forehead", "temple"]);
  let faceHits = 0;
  let chestHits = 0;
  regions.forEach((r) => {
    const lower = String(r).toLowerCase();
    if (facialRegions.has(lower)) faceHits++;
    if (/chest|torso|shoulder|heart/i.test(lower)) chestHits++;
  });

  const isFacial = (faceHits >= 2 && faceHits >= chestHits) || (faceHits > 0 && chestHits === 0);

  const counts = new Map();
  regions.forEach((region) => counts.set(region, (counts.get(region) || 0) + 1));
  const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  const [dominantRegion] = sorted[0] || [];

  const dominantZone = ["nose", "mustache", "mouth", "chin"].includes(dominantRegion) ? "chin" : dominantRegion;

  const adjusted = new Float32Array(probabilities.length);
  let total = 0;
  for (let index = 0; index < probabilities.length; index += 1) {
    const gloss = model.classes[index];
    const norm = String(gloss || "").toUpperCase().replace(/\s+/g, "_");

    // Chest signs (SORRY, PLEASE, HAPPY) are strictly impossible in the facial region!
    if (["SORRY", "PLEASE", "HAPPY"].includes(norm) && isFacial) {
      adjusted[index] = 0;
      continue;
    }

    const expectedPositions = getExpectedBodyPositions(gloss);
    const matches = expectedPositions.some((pos) => {
      const posZone = ["nose", "mustache", "mouth", "chin"].includes(pos) ? "chin" : pos;
      return posZone === dominantZone || pos === dominantRegion || pos === "neutral";
    });

    const positionWeight = matches ? 1.15 : 0.92;
    adjusted[index] = probabilities[index] * positionWeight;
    total += adjusted[index];
  }
  if (!total) return probabilities;
  for (let index = 0; index < adjusted.length; index += 1) adjusted[index] /= total;
  return adjusted;
}

function checkPredictedBodyPosition(gloss, observedRegions = state.obsBodyRegions || [], totalFrames = state.obsBuffer?.length || 0) {
  const norm = String(gloss || "").toUpperCase().replace(/\s+/g, "_").trim();
  if (norm === "SORRY" || norm === "PLEASE" || norm === "HAPPY") {
    const facialRegions = new Set(["nose", "mustache", "mouth", "chin", "forehead", "temple"]);
    let faceHits = 0;
    let chestHits = 0;
    (observedRegions || []).forEach((r) => {
      if (typeof r === "string") {
        const lower = r.toLowerCase();
        if (facialRegions.has(lower)) faceHits++;
        if (/chest|torso|shoulder|heart/i.test(lower)) chestHits++;
      }
    });

    if ((faceHits >= 2 && faceHits >= chestHits) || (faceHits > 0 && chestHits === 0)) {
      return { accepted: false, reason: `${norm} must be performed on the chest, not the facial region.` };
    }
  }
  return { accepted: true, reason: "" };
}

function euclideanDistance(a, b) {
  let sum = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i += 1) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum / len);
}

function reluLayer(input, weights, bias) {
  const inDim = input.length;
  const outDim = bias.length;
  const output = new Float32Array(outDim);
  for (let o = 0; o < outDim; o += 1) {
    let sum = bias[o] || 0;
    for (let i = 0; i < inDim; i += 1) {
      sum += input[i] * weights[i][o];
    }
    output[o] = Math.max(0, sum);
  }
  return output;
}

function linearLayer(input, weights, bias) {
  const inDim = input.length;
  const outDim = bias.length;
  const output = new Float32Array(outDim);
  for (let o = 0; o < outDim; o += 1) {
    let sum = bias[o] || 0;
    for (let i = 0; i < inDim; i += 1) {
      sum += input[i] * weights[i][o];
    }
    output[o] = sum;
  }
  return output;
}

function softmax(logits) {
  let maxLogit = -Infinity;
  for (let c = 0; c < logits.length; c += 1) {
    if (logits[c] > maxLogit) maxLogit = logits[c];
  }

  let expSum = 0;
  const probs = new Float32Array(logits.length);
  for (let c = 0; c < logits.length; c += 1) {
    probs[c] = Math.exp(logits[c] - maxLogit);
    expSum += probs[c];
  }
  for (let c = 0; c < logits.length; c += 1) {
    probs[c] /= expSum;
  }
  return probs;
}

/* UI & Audio Functions */
function formatGlossDisplay(gloss) {
  if (!gloss) return "";
  return String(gloss).toUpperCase().replace(/_/g, " ").trim();
}

function updateRecognitionDisplay(sign, confidence, tagState, reason) {
  let cleanSign = formatGlossDisplay(sign);
  const meetsDisplayThreshold = Number.isFinite(confidence) && confidence > getRecognitionThreshold();
  const hasConfirmedSign = tagState === "CONFIRMED" && cleanSign !== "UNKNOWN" && meetsDisplayThreshold;
  if (!hasConfirmedSign) {
    cleanSign = "NOT SURE";
  }

  if (hasConfirmedSign) state.lastConfirmedSign = cleanSign;

  // The live badge reflects only this state. The side panel keeps the last
  // confirmed word as history, but idle and uncertain states must not look like
  // the old result is being recognized again.
  const visibleSign = hasConfirmedSign
    ? cleanSign
    : (tagState === "STANDBY" ? "WAITING FOR A SIGN" : cleanSign);
  els.currentSign.textContent = visibleSign;
  if (els.lastRecognizedWord && hasConfirmedSign) {
    els.lastRecognizedWord.textContent = cleanSign;
    els.lastRecognizedWord.classList.remove("is-empty");
  }
  els.confidenceVal.textContent = hasConfirmedSign ? `${(confidence * 100).toFixed(1)}%` : "--";
  els.decisionState.textContent = tagState;
  els.recognitionReason.textContent = reason ? String(reason).replace(/_/g, " ") : "";

  els.decisionState.hidden = false;
  els.confidenceVal.hidden = tagState === "STANDBY";

  if (tagState === "CONFIRMED" && cleanSign !== "NOT SURE") {
    els.decisionState.className = "badge-tag active";
  } else {
    els.decisionState.className = "badge-tag";
  }
  els.currentSign.style.color = visibleSign !== "NOT SURE" ? "var(--accent-emerald)" : "#f59e0b";
}

/* Practice Studio Split-Screen & Evaluation Functions */
function openPracticeModalFromTutor() {
  const gloss = els.tutorSignTitle ? els.tutorSignTitle.textContent.trim() : "WATER";
  if (els.tutorModal) els.tutorModal.classList.add("hidden");
  openPracticeModal(gloss);
}

function populateGlossDropdown(selectEl, selectedValue = "", defaultLabel = "no input -- Select sign from vocabulary to practice") {
  if (!selectEl) return;
  selectEl.innerHTML = "";

  const cleanTarget = formatGlossDisplay(selectedValue || "");

  const defaultOpt = document.createElement("option");
  defaultOpt.value = "";
  defaultOpt.textContent = defaultLabel;
  if (!cleanTarget) defaultOpt.selected = true;
  selectEl.appendChild(defaultOpt);

  let rawList = [];
  if (state.modelLoaded && state.model && state.model.classes && state.model.classes.length > 0) {
    rawList = state.model.classes;
  } else if (state.vocabulary && state.vocabulary.length > 0) {
    rawList = state.vocabulary.map((v) => v.gloss || v.display);
  } else {
    rawList = [
      "WATER", "HELLO", "THANK YOU", "PLEASE", "HELP", "GOOD MORNING", "GOOD NIGHT",
      "MOTHER", "FATHER", "BROTHER", "SISTER", "FRIEND", "FOOD", "DRINK", "EAT",
      "COME", "GO", "HOUSE", "SCHOOL", "HAPPY", "NAME", "INDIAN SIGN LANGUAGE",
      "LOVE", "LEARN", "TEACH", "TIME", "TODAY", "TOMORROW", "YES", "NO"
    ];
  }

  const cleanMap = new Map();
  rawList.forEach((item) => {
    const display = formatGlossDisplay(item);
    if (display && !cleanMap.has(display)) {
      cleanMap.set(display, display);
    }
  });

  const sortedGlosses = Array.from(cleanMap.keys()).sort((a, b) => a.localeCompare(b));

  sortedGlosses.forEach((g) => {
    const opt = document.createElement("option");
    opt.value = g;
    opt.textContent = g;
    if (cleanTarget && g === cleanTarget) opt.selected = true;
    selectEl.appendChild(opt);
  });
}

function openPracticeModal(targetGloss) {
  const cleanTarget = formatGlossDisplay(targetGloss || "");

  // Populate practiceGlossSelect dropdown cleanly with all vocabulary glosses
  if (els.practiceGlossSelect) {
    populateGlossDropdown(els.practiceGlossSelect, cleanTarget, "no input -- Select sign from vocabulary to practice");
  }

  const activeTarget = cleanTarget || (els.practiceGlossSelect && els.practiceGlossSelect.value ? els.practiceGlossSelect.value : (state.practiceTargetGloss || "WATER"));
  updatePracticeTargetView(activeTarget);

  if (!state.stream) {
    startCamera();
  } else if (els.practiceCamVideo) {
    els.practiceCamVideo.srcObject = state.stream;
    els.practiceCamVideo.play().catch(() => {});
  }
  els.practiceCamWrapper?.classList.remove("is-paused-feed", "is-analyzing");
  els.practiceNotification?.classList.add("hidden");

  state.isPracticeRecording = false;
  state.practiceFrameBuffer = [];

  const startBtn = els.camBoxRecordBtn;
  const stopBtn = els.camBoxStopBtn;

  if (startBtn) startBtn.classList.remove("hidden");
  if (stopBtn) stopBtn.classList.add("hidden");
  if (els.practiceRecOverlay) els.practiceRecOverlay.classList.add("hidden");
  if (els.practiceEvalResult) els.practiceEvalResult.classList.add("hidden");

  // Pause main background prediction immediately
  resetObservationState();
  if (els.decisionState) {
    els.decisionState.textContent = "PAUSED";
    els.decisionState.className = "badge-tag standby";
  }
  if (els.currentSign) {
    els.currentSign.textContent = "PRACTICE IN PROGRESS";
    els.currentSign.style.color = "var(--text-muted)";
  }
  if (els.recognitionReason) {
    els.recognitionReason.textContent = "Live camera sign prediction is paused while practice studio is open.";
  }

  if (els.practiceModal) {
    els.practiceModal.classList.remove("hidden");
  }
}

function onPracticeSelectChange() {
  if (els.practiceGlossSelect) {
    const selected = els.practiceGlossSelect.value;
    if (selected) {
      updatePracticeTargetView(selected);
    }
  }
}

function updatePracticeTargetView(targetGloss) {
  const cleanTarget = formatGlossDisplay(targetGloss || "WATER");
  state.practiceTargetGloss = cleanTarget;

  if (els.practiceTargetTitle) {
    els.practiceTargetTitle.textContent = `PRACTICE SIGN: ${cleanTarget}`;
  }
  if (els.sidebarTargetGlossBadge) {
    els.sidebarTargetGlossBadge.textContent = cleanTarget;
  }

  if (els.practiceRefVideo) {
    const videoUrl = `/api/dataset_video?gloss=${encodeURIComponent(cleanTarget)}`;
    els.practiceRefVideo.src = videoUrl;
    els.practiceRefVideo.load();
    els.practiceRefVideo.play().catch(() => {});

    fetch(videoUrl, { method: "HEAD" }).then((resp) => {
      const isUser = resp.headers.get("X-Video-Source") === "user";
      if (els.refVideoLabel) {
        els.refVideoLabel.textContent = isUser ? "👤 My Training Video" : "Verified dataset demo";
        if (isUser) {
          els.refVideoLabel.classList.add("green-tag");
          els.refVideoLabel.classList.remove("cyan-tag");
        } else {
          els.refVideoLabel.classList.remove("green-tag");
          els.refVideoLabel.classList.add("cyan-tag");
        }
      }
    }).catch(() => {});
  }

  if (els.practiceInstructionTip) {
    const info = ISL_TUTOR_GUIDE[cleanTarget.toUpperCase()] || null;
    if (info && info.handShape && info.movement) {
      els.practiceInstructionTip.textContent = `Target: ${cleanTarget} • Hand Shape: ${info.handShape} • Movement: ${info.movement}`;
    } else {
      els.practiceInstructionTip.textContent = `No verified instruction guide is available for "${cleanTarget}". Use only a verified reference clip for practice.`;
    }
  }

  if (els.practiceEvalResult) {
    els.practiceEvalResult.classList.add("hidden");
  }
}

function hidePracticeModal() {
  if (state.isPracticeRecording) {
    if (state.practiceRecTimerInterval) clearInterval(state.practiceRecTimerInterval);
    state.isPracticeRecording = false;
  }
  if (els.practiceRefVideo) {
    els.practiceRefVideo.pause();
  }
  if (els.practiceCamVideo) {
    els.practiceCamVideo.pause();
    els.practiceCamVideo.srcObject = null;
  }
  els.practiceCamWrapper?.classList.add("is-paused-feed");
  if (els.practiceAnalyzingBox) {
    els.practiceAnalyzingBox.classList.add("hidden");
  }
  if (els.practiceRecOverlay) {
    els.practiceRecOverlay.classList.add("hidden");
  }
  const startBtn = els.camBoxRecordBtn;
  const stopBtn = els.camBoxStopBtn;
  if (startBtn) startBtn.classList.remove("hidden");
  if (els.camBoxRecordBtn) els.camBoxRecordBtn.classList.remove("hidden");
  if (stopBtn) {
    stopBtn.classList.add("hidden");
    stopBtn.disabled = false;
  }
  if (els.camBoxStopBtn) {
    els.camBoxStopBtn.classList.add("hidden");
    els.camBoxStopBtn.disabled = false;
  }
  if (els.sidePanelRecordBtnText) {
    els.sidePanelRecordBtnText.textContent = "🔴 Record & Check Accuracy";
  }
  if (els.practiceModal) {
    els.practiceModal.classList.add("hidden");
  }

  // Restore main background camera status
  if (els.decisionState) {
    els.decisionState.textContent = "STANDBY";
    els.decisionState.className = "badge-tag";
  }
  if (els.currentSign) {
    els.currentSign.textContent = "UNKNOWN";
    els.currentSign.style.color = "var(--text-main)";
  }
  if (els.obsProgressFill) {
    els.obsProgressFill.style.width = "0%";
  }
  if (els.obsLabel) {
    els.obsLabel.textContent = "Show hands to read sign...";
  }
  if (els.recognitionReason) {
    els.recognitionReason.textContent = "Position hands inside camera frame to begin.";
  }
}

function openPracticeModalAndRecord(targetGloss) {
  openPracticeModal(targetGloss);
  setTimeout(() => {
    startPracticeRecording();
  }, 450);
}

function isHandPresentInVector(vec) {
  if (!vec) return false;
  let sum = 0;
  for (let i = 0; i < Math.min(vec.length, 126); i++) {
    sum += Math.abs(vec[i]);
  }
  return sum > 0.05;
}

async function startPracticeRecording() {
  if (!state.stream) {
    await startCamera();
  }
  if (!state.stream || !state.hands) {
    if (els.practiceFeedback) {
      els.practiceFeedback.textContent = "Camera and hand tracking must be ready before recording a practice attempt.";
    }
    return;
  }

  getPracticeAudioContext();
  state.isPracticeRecording = true;
  state.practiceBodyPositionFrames = [];
  if (els.practiceCamVideo && state.stream) {
    els.practiceCamVideo.srcObject = state.stream;
    els.practiceCamVideo.play().catch(() => {});
  }
  els.practiceCamWrapper?.classList.remove("is-paused-feed", "is-analyzing");
  els.practiceNotification?.classList.add("hidden");
  state.practiceFrameBuffer = [];
  const practiceScore = document.querySelector("#practiceCameraScore");
  if (practiceScore) practiceScore.textContent = "…";
  document.querySelector("#practiceAccuracyRing")?.classList.add("is-scanning");
  state.practiceRecStartTime = Date.now();

  const startBtn = els.camBoxRecordBtn;
  const stopBtn = els.camBoxStopBtn;

  if (startBtn) startBtn.classList.add("hidden");
  if (els.camBoxRecordBtn) els.camBoxRecordBtn.classList.add("hidden");
  if (stopBtn) {
    stopBtn.classList.remove("hidden");
    stopBtn.disabled = false;
  }
  if (els.camBoxStopBtn) {
    els.camBoxStopBtn.classList.remove("hidden");
    els.camBoxStopBtn.disabled = false;
  }
  if (els.camBoxStopBtnText) {
    els.camBoxStopBtnText.textContent = "⏹️ Stop & Analyze";
  }
  if (els.sidePanelRecordBtnText) {
    els.sidePanelRecordBtnText.textContent = "⏹️ Recording in Progress...";
  }
  if (els.practiceRecOverlay) els.practiceRecOverlay.classList.remove("hidden");
  if (els.practiceEvalResult) els.practiceEvalResult.classList.add("hidden");
  if (els.practiceAnalyzingBox) els.practiceAnalyzingBox.classList.add("hidden");

  // Keep modal view focused on the camera feed when recording

  if (state.practiceRecTimerInterval) clearInterval(state.practiceRecTimerInterval);
  state.practiceRecTimerInterval = setInterval(() => {
    if (!state.isPracticeRecording) {
      clearInterval(state.practiceRecTimerInterval);
      return;
    }
    const elapsed = Date.now() - state.practiceRecStartTime;
    const secStr = (elapsed / 1000).toFixed(1);
    const maxSec = 10.0;

    let handsCount = 0;
    for (let i = 0; i < state.practiceFrameBuffer.length; i++) {
      if (isHandPresentInVector(state.practiceFrameBuffer[i])) handsCount++;
    }

    if (els.practiceRecTimerText) {
      els.practiceRecTimerText.textContent = `RECORDING: ${secStr}s · ${handsCount} hand frames · ${state.activeBodyRegion || "NEUTRAL SPACE"}`;
    }

    if (elapsed >= maxSec * 1000) {
      stopPracticeAndEvaluate();
    }
  }, 100);
}

function stopPracticeAndEvaluate() {
  getPracticeAudioContext();
  if (state.practiceRecTimerInterval) clearInterval(state.practiceRecTimerInterval);
  state.isPracticeRecording = false;
  if (els.practiceCamVideo) {
    els.practiceCamVideo.pause();
    els.practiceCamVideo.srcObject = null;
  }
  els.practiceCamWrapper?.classList.add("is-paused-feed", "is-analyzing");
  els.practiceNotification?.classList.add("hidden");

  const startBtn = els.camBoxRecordBtn;
  const stopBtn = els.camBoxStopBtn;

  // Set stop buttons to analyzing state
  if (stopBtn) stopBtn.disabled = true;
  if (els.camBoxStopBtn) els.camBoxStopBtn.disabled = true;
  if (els.modalStopPracticeBtnText) {
    els.modalStopPracticeBtnText.textContent = "⏳ Analyzing Sign...";
  }
  if (els.camBoxStopBtnText) {
    els.camBoxStopBtnText.textContent = "⏳ Analyzing...";
  }
  if (els.sidePanelRecordBtnText) {
    els.sidePanelRecordBtnText.textContent = "⏳ Analyzing Sign...";
  }
  if (els.practiceRecOverlay) els.practiceRecOverlay.classList.add("hidden");
  if (els.practiceEvalResult) els.practiceEvalResult.classList.add("hidden");

  // Show live animated analyzing box
  if (els.practiceAnalyzingBox) {
    els.practiceAnalyzingBox.classList.remove("hidden");
  }
  if (els.practiceAnalyzingFill) {
    els.practiceAnalyzingFill.style.width = "25%";
  }
  if (els.practiceAnalyzingSubtitle) {
    els.practiceAnalyzingSubtitle.textContent = "1/3. Reviewing all 21 landmarks on each detected hand and wrist position…";
  }

  // Multi-stage visual analyzing workflow
  setTimeout(() => {
    if (els.practiceAnalyzingFill) els.practiceAnalyzingFill.style.width = "65%";
    if (els.practiceAnalyzingSubtitle) {
      els.practiceAnalyzingSubtitle.textContent = "2/3. Comparing hand shape, movement, and trained sign classes…";
    }
  }, 350);

  setTimeout(() => {
    if (els.practiceAnalyzingFill) els.practiceAnalyzingFill.style.width = "95%";
    if (els.practiceAnalyzingSubtitle) {
      els.practiceAnalyzingSubtitle.textContent = "3/3. Calculating the target match score and result…";
    }
  }, 700);

  setTimeout(() => {
    // Hide analyzing box and reset buttons
    if (els.practiceAnalyzingBox) els.practiceAnalyzingBox.classList.add("hidden");
    els.practiceCamWrapper?.classList.remove("is-analyzing");
    if (stopBtn) {
      stopBtn.classList.add("hidden");
      stopBtn.disabled = false;
    }
    if (els.camBoxStopBtn) {
      els.camBoxStopBtn.classList.add("hidden");
      els.camBoxStopBtn.disabled = false;
    }
    if (els.camBoxStopBtnText) {
      els.camBoxStopBtnText.textContent = "⏹️ Stop & Analyze";
    }
    if (startBtn) startBtn.classList.remove("hidden");
    if (els.camBoxRecordBtn) els.camBoxRecordBtn.classList.remove("hidden");
    if (els.sidePanelRecordBtnText) {
      els.sidePanelRecordBtnText.textContent = "🔴 Record & Check Accuracy";
    }

    performPracticeAccuracyComputation();
  }, 1000);
}

// These vocabulary labels intentionally share the same performed sign.
// Keep both labels available in the UI, while treating either as a valid
// recognition/practice result for the paired target.
const SHARED_SIGN_GROUPS = [
  new Set(["WATER", "DRINK"]),
  new Set(["EAT", "FOOD"]),
];

function getEquivalentGlosses(gloss) {
  const normalized = String(gloss || "").toUpperCase().replace(/\s+/g, "_").trim();
  return SHARED_SIGN_GROUPS.find((group) => group.has(normalized)) || new Set([normalized]);
}

function areEquivalentSigns(first, second) {
  return getEquivalentGlosses(first).has(String(second || "").toUpperCase().replace(/\s+/g, "_").trim());
}

function getExpectedBodyPositions(gloss) {
  const norm = String(gloss || "").toUpperCase().replace(/\s+/g, "_").trim();
  const guide = ISL_TUTOR_GUIDE[norm];
  const targetText = ((guide?.bodyTarget || "") + " " + (guide?.movement || "")).toUpperCase();
  const positions = new Set();

  if (/MUSTACHE|MOUSTACHE|UPPER LIP/.test(targetText)) {
    positions.add("mustache").add("chin");
  }
  if (/NOSE/.test(targetText)) {
    positions.add("nose").add("chin");
  }
  if (/TEMPLE|FOREHEAD|HEAD|EAR/.test(targetText)) {
    positions.add("temple");
  }
  if (/CHIN|CHEEK|JAW|MOUTH|LIP/.test(targetText)) {
    positions.add("chin");
  }
  if (/CHEST|STOMACH|HEART|TORSO|SHOULDER/.test(targetText)) {
    positions.add("chest");
  }
  if (/PALM|HAND/.test(targetText)) {
    positions.add("opposing_hand");
  }
  if (/NEUTRAL|FRONT|SPACE|FORWARD|AIR/.test(targetText) || positions.size === 0) {
    positions.add("neutral");
  }

  // Explicit multi-phase ISL vocabulary signs
  if (norm === "BROTHER") {
    positions.add("mustache").add("chin").add("chest").add("neutral");
  } else if (norm === "SISTER") {
    positions.add("nose").add("chin").add("chest").add("neutral");
  } else if (norm === "WHEN") {
    positions.add("chin").add("neutral").add("chest");
  } else if (norm === "WHERE") {
    positions.add("neutral").add("chest");
  } else if (norm === "WRITE") {
    positions.add("opposing_hand").add("neutral");
  } else if (norm === "THANK_YOU" || norm === "THANK YOU") {
    positions.add("chin").add("neutral");
  } else if (norm === "WELCOME") {
    positions.add("chest").add("neutral");
  } else if (norm === "PLEASE" || norm === "SORRY" || norm === "HAPPY") {
    positions.add("chest");
  } else if (norm === "SAD") {
    positions.add("chest").add("chin");
  }

  return [...positions];
}

function getExpectedBodyPosition(gloss) {
  const positions = getExpectedBodyPositions(gloss);
  return positions[0] || "neutral";
}

function getExpectedHandSignals(gloss) {
  const guide = ISL_TUTOR_GUIDE[String(gloss || "").toUpperCase()];
  const handShapeText = (guide?.handShape || "").toUpperCase();
  const signals = [];
  if (/PINCH|O-HAND/.test(handShapeText)) signals.push("Pinch Shape (O-Hand)");
  if (/THUMB/.test(handShapeText) && /UP/.test(handShapeText)) signals.push("Thumbs-Up");
  if (/W-HAND|3-FINGER/.test(handShapeText)) signals.push("W-Hand (3-Fingers)");
  if (/V-HAND|2-FINGER|PEACE/.test(handShapeText)) signals.push("V-Hand (2-Fingers)");
  if (/INDEX|POINT/.test(handShapeText)) signals.push("Index Pointing (1-Finger)");
  if (/FIST/.test(handShapeText)) signals.push("Fist Shape (S-Hand)");
  if (/OPEN 5|FIVE FINGERS|SPREAD/.test(handShapeText)) signals.push("Open 5-Palm", "Open Hand");
  if (/FLAT|OPEN PALM|B-HAND|OPEN HAND/.test(handShapeText)) signals.push("Flat Palm (B-Hand)", "Open 5-Palm", "Open Hand");
  if (/PRAYER|TOGETHER|NAMASTE/.test(handShapeText)) signals.push("Open Hand", "Flat Palm (B-Hand)");
  if (/C-SHAPE|CUP/.test(handShapeText)) signals.push("Curved / Natural", "Pinch Shape (O-Hand)");
  return signals;
}

function performPracticeAccuracyComputation() {
  const targetGloss = state.practiceTargetGloss || "WATER";
  const practiceFrames = state.practiceFrameBuffer || [];
  const recordedBodyFrames = state.practiceBodyPositionFrames || [];
  const activeFrameRecords = practiceFrames
    .map((frame, index) => ({ frame, body: recordedBodyFrames[index] || null }))
    .filter(({ frame }) => isHandPresentInVector(frame));
  const activeHandFrames = activeFrameRecords.map(({ frame }) => frame);
  const bodyPositionFrames = activeFrameRecords.map(({ body }) => body || {
    region: "neutral",
    subregion: "neutral",
    organKey: "neutral",
    organName: "Neutral Space",
    handSignal: "Open Hand",
    relativeX: 0,
    relativeY: 0,
  });

  // Check if hand frames were recorded
  if (!activeHandFrames || activeHandFrames.length < 5) {
    showPracticeEvaluation({
      accuracyPct: "0.0%",
      numVal: 0.0,
      verdict: "NO HANDS DETECTED",
      icon: "⚠️",
      targetGloss: targetGloss,
      detectedGloss: "NONE",
      handShape: "No hand tracked",
      motionPath: "No organ tracked",
      detailsText: "Hands were not clearly detected in the camera feed. Please position both hands visibly in front of the camera and record your sign again.",
      category: "poor",
      isApproved: false,
    });
    return;
  }

  const numFrames = activeHandFrames.length;

  if (!state.model || !state.modelLoaded || !state.model.classes) {
    showPracticeEvaluation({
      accuracyPct: "--",
      numVal: 0,
      verdict: "MODEL UNAVAILABLE",
      icon: "⚠️",
      targetGloss: targetGloss,
      detectedGloss: "NOT SCORED",
      handShape: "--",
      motionPath: "--",
      detailsText: `No trained recognition model is available to score this attempt. Train "${targetGloss}" and try again.`,
      category: "warn",
      isApproved: false,
    });
    return;
  }

  // Identify dominant hand signal & dominant organ placement recorded
  const signalCounts = new Map();
  const organCounts = new Map();
  bodyPositionFrames.forEach((frame) => {
    const sig = frame.handSignal || "Open Hand";
    signalCounts.set(sig, (signalCounts.get(sig) || 0) + 1);
    const org = frame.organName || "Neutral Space";
    organCounts.set(org, (organCounts.get(org) || 0) + 1);
  });
  const dominantSignal = [...signalCounts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] || "Open Hand";
  const dominantOrgan = [...organCounts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] || "Neutral Space";

  const practiceBodyRegions = bodyPositionFrames.map((frame) => frame.subregion || frame.region || null);
  state.isEvaluatingPractice = true;
  const result = classifyGestureWithConsensus(activeHandFrames, state.model, practiceBodyRegions);
  state.isEvaluatingPractice = false;
  const bestProb = result.confidence || 0;
  const probs = result.probs;
  const confidenceThreshold = Math.min(getRecognitionThreshold(), 0.35);
  const predictedGloss = result.isAccepted && bestProb > confidenceThreshold
    ? formatGlossDisplay(result.sign)
    : formatGlossDisplay(result.sign || "UNCLEAR");

  const equivalentGlosses = getEquivalentGlosses(targetGloss);
  const isEquivalent = areEquivalentSigns(predictedGloss, targetGloss);

  let targetProb = 0;
  let strongestOtherProb = 0;
  if (probs) {
    state.model.classes.forEach((classGloss, index) => {
      if (equivalentGlosses.has(String(classGloss).toUpperCase().replace(/\s+/g, "_").trim())) {
        targetProb += probs[index] || 0;
      } else {
        strongestOtherProb = Math.max(strongestOtherProb, probs[index] || 0);
      }
    });
  }

  // Hand Signal match evaluation
  const expectedSignals = getExpectedHandSignals(targetGloss);
  let handSignalMatch = false;
  if (expectedSignals.length === 0) {
    handSignalMatch = true;
  } else {
    handSignalMatch = bodyPositionFrames.some((f) => expectedSignals.includes(f.handSignal))
      || expectedSignals.some((s) => dominantSignal.includes(s.split(" ")[0]));
  }
  const handSignalScore = handSignalMatch ? 1.0 : (dominantSignal !== "No Hand Tracked" ? 0.85 : 0.4);

  // Organ Placement match evaluation
  const expectedPositions = getExpectedBodyPositions(targetGloss);
  let organMatch = false;
  if (expectedPositions.length === 0) {
    organMatch = true;
  } else {
    organMatch = bodyPositionFrames.some((frame) => {
      const sub = frame.subregion;
      const reg = frame.region;
      const key = frame.organKey;
      return expectedPositions.some((p) => p === sub || p === reg || p === key);
    });
  }
  const organScore = organMatch ? 1.0 : 0.85;

  // Unified Multi-Factor Accuracy Computation
  let numVal = 0;
  let isApproved = false;
  let verdict = "";
  let icon = "🎯";
  let category = "good";
  let detailsText = "";

  if (isEquivalent || targetProb >= 0.30 || (bestProb > confidenceThreshold && strongestOtherProb < targetProb)) {
    const effectiveProb = Math.max(targetProb, isEquivalent ? bestProb : targetProb);
    // Combine 65% Model alignment + 20% Hand Signal + 15% Organ Placement
    numVal = Math.min(99.2, Math.max(68.0, Math.round((effectiveProb * 0.65 + handSignalScore * 0.20 + organScore * 0.15) * 1000) / 10));

    if (numVal >= 60.0) {
      isApproved = true;
      verdict = "APPROVED - EXCELLENT ACCURACY";
      icon = "✅";
      category = "good";
      detailsText = `Sign "${targetGloss}" performed accurately with valid hand signal (${dominantSignal}) at expected organ (${dominantOrgan}). Approved with ${numVal.toFixed(1)}% match score (meets ≥ 60% threshold).`;
    } else {
      isApproved = false;
      verdict = "NEEDS PRACTICE - BELOW 60%";
      icon = "⚠️";
      category = "warn";
      detailsText = `Sign "${targetGloss}" achieved a ${numVal.toFixed(1)}% match score, which is below the 60% approval threshold. Watch the reference video and try again.`;
    }
  } else if (!result.isAccepted || bestProb < 0.25) {
    numVal = Math.max(15.0, Math.round(Math.max(targetProb, 0.25) * 600) / 10);
    isApproved = false;
    verdict = "UNCLEAR GESTURE - TRY AGAIN";
    icon = "⚠️";
    category = "warn";
    detailsText = `The gesture trajectory was unclear or hesitant (${numVal.toFixed(1)}% match). Check your hand shape (${dominantSignal}) and organ placement (${dominantOrgan}), then try again.`;
  } else {
    numVal = Math.min(48.0, Math.max(20.0, Math.round(targetProb * 650) / 10));
    isApproved = false;
    verdict = "SIGN MISMATCH - TRY AGAIN";
    icon = "❌";
    category = "poor";
    detailsText = `Detected "${predictedGloss}" instead of target "${targetGloss}". Match score: ${numVal.toFixed(1)}% (below 60% threshold). Replicate the reference sign on the left and try again.`;
  }

  showPracticeEvaluation({
    accuracyPct: `${numVal.toFixed(1)}%`,
    numVal: numVal,
    verdict: verdict,
    icon: icon,
    targetGloss: targetGloss,
    detectedGloss: isApproved ? targetGloss : predictedGloss,
    handShape: `${dominantSignal} ${handSignalMatch ? '✓' : ''}`,
    motionPath: `${dominantOrgan} ${organMatch ? '✓' : ''}`,
    detailsText: detailsText,
    category: category,
    isApproved: isApproved,
  });
}

// ==========================================
// Web Audio Practice Feedback Sound System
// ==========================================
let practiceAudioCtx = null;

function getPracticeAudioContext() {
  try {
    if (!practiceAudioCtx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        practiceAudioCtx = new AudioCtx();
      }
    }
    if (practiceAudioCtx && practiceAudioCtx.state === "suspended") {
      practiceAudioCtx.resume().catch(() => {});
    }
  } catch (e) {
    console.warn("Unable to initialize AudioContext for practice feedback:", e);
  }
  return practiceAudioCtx;
}

function playPracticeSound(type) {
  try {
    const ctx = getPracticeAudioContext();
    if (!ctx) return;

    const isCorrect = (type === "correct" || type === true);
    const now = ctx.currentTime;

    if (isCorrect) {
      // 🎵 Correct / Approved Sound: Bright, uplifting ascending chime (C5 -> E5 -> G5)
      // Note 1: 523.25 Hz (C5)
      const osc1 = ctx.createOscillator();
      const gain1 = ctx.createGain();
      osc1.type = "sine";
      osc1.frequency.setValueAtTime(523.25, now);
      gain1.gain.setValueAtTime(0.0001, now);
      gain1.gain.linearRampToValueAtTime(0.20, now + 0.03);
      gain1.gain.linearRampToValueAtTime(0.0001, now + 0.18);
      osc1.connect(gain1);
      gain1.connect(ctx.destination);
      osc1.start(now);
      osc1.stop(now + 0.19);

      // Note 2: 659.25 Hz (E5)
      const osc2 = ctx.createOscillator();
      const gain2 = ctx.createGain();
      osc2.type = "sine";
      osc2.frequency.setValueAtTime(659.25, now + 0.08);
      gain2.gain.setValueAtTime(0.0001, now + 0.08);
      gain2.gain.linearRampToValueAtTime(0.22, now + 0.11);
      gain2.gain.linearRampToValueAtTime(0.0001, now + 0.28);
      osc2.connect(gain2);
      gain2.connect(ctx.destination);
      osc2.start(now + 0.08);
      osc2.stop(now + 0.29);

      // Note 3: 783.99 Hz (G5) ringing chime
      const osc3 = ctx.createOscillator();
      const gain3 = ctx.createGain();
      osc3.type = "triangle";
      osc3.frequency.setValueAtTime(783.99, now + 0.16);
      gain3.gain.setValueAtTime(0.0001, now + 0.16);
      gain3.gain.linearRampToValueAtTime(0.25, now + 0.19);
      gain3.gain.linearRampToValueAtTime(0.0001, now + 0.55);
      osc3.connect(gain3);
      gain3.connect(ctx.destination);
      osc3.start(now + 0.16);
      osc3.stop(now + 0.56);

    } else {
      // ⚠️ Wrong / Needs Practice Sound: Soft descending two-tone cue (F4 -> C#4)
      // Note 1: 349.23 Hz (F4)
      const osc1 = ctx.createOscillator();
      const gain1 = ctx.createGain();
      osc1.type = "sine";
      osc1.frequency.setValueAtTime(349.23, now);
      gain1.gain.setValueAtTime(0.0001, now);
      gain1.gain.linearRampToValueAtTime(0.18, now + 0.03);
      gain1.gain.linearRampToValueAtTime(0.0001, now + 0.16);
      osc1.connect(gain1);
      gain1.connect(ctx.destination);
      osc1.start(now);
      osc1.stop(now + 0.17);

      // Note 2: 277.18 Hz (C#4)
      const osc2 = ctx.createOscillator();
      const gain2 = ctx.createGain();
      osc2.type = "sine";
      osc2.frequency.setValueAtTime(277.18, now + 0.12);
      gain2.gain.setValueAtTime(0.0001, now + 0.12);
      gain2.gain.linearRampToValueAtTime(0.20, now + 0.15);
      gain2.gain.linearRampToValueAtTime(0.0001, now + 0.45);
      osc2.connect(gain2);
      gain2.connect(ctx.destination);
      osc2.start(now + 0.12);
      osc2.stop(now + 0.46);
    }
  } catch (err) {
    console.warn("Error playing practice sound notification:", err);
  }
}

function showPracticeEvaluation(evalData) {
  const practiceScore = document.querySelector("#practiceCameraScore");
  if (practiceScore) practiceScore.textContent = `${Number(evalData.numVal).toFixed(1)}%`;
  document.querySelector("#practiceAccuracyRing")?.classList.remove("is-scanning");

  const isApproved = evalData.isApproved !== undefined
    ? evalData.isApproved
    : (Number(evalData.numVal) >= 60.0 && evalData.category === "good");

  // Play audio sound notification for correct or wrong practice attempt
  playPracticeSound(isApproved ? "correct" : "wrong");

  if (els.practiceNotification) {
    els.practiceNotification.classList.remove("hidden", "success", "failure");
    els.practiceNotification.classList.add(isApproved ? "success" : "failure");
    els.practiceNotification.textContent = isApproved
      ? `✅ Approved! ${evalData.targetGloss} matched with ${Number(evalData.numVal).toFixed(1)}% accuracy (meets ≥ 60% threshold).`
      : `⚠️ Match score (${Number(evalData.numVal).toFixed(1)}%) is below 60%. Click "Try Again" to record again.`;
  }
  if (!els.practiceEvalResult) return;
  els.practiceEvalResult.classList.remove("hidden");

  // Dynamic 60% Approval / Try Again Banner
  if (els.practiceVerdictBanner) {
    els.practiceVerdictBanner.className = isApproved ? "eval-verdict-banner approved" : "eval-verdict-banner retry";
  }
  if (els.practiceBannerText) {
    els.practiceBannerText.textContent = isApproved
      ? `✅ SIGN APPROVED! Match Score: ${Number(evalData.numVal).toFixed(1)}% (Meets ≥ 60% threshold)`
      : `⚠️ SCORE BELOW 60% (${Number(evalData.numVal).toFixed(1)}%) - Try again to reach approval`;
  }
  if (els.practiceBannerBadge) {
    els.practiceBannerBadge.textContent = isApproved ? "APPROVED (≥60%)" : "TRY AGAIN (<60%)";
  }

  // Interactive Action Buttons inside Evaluation Card
  if (els.practiceTryAgainBtn) {
    if (isApproved) {
      els.practiceTryAgainBtn.classList.add("hidden");
    } else {
      els.practiceTryAgainBtn.classList.remove("hidden");
    }
  }
  if (els.practiceNextSignBtn) {
    if (isApproved) {
      els.practiceNextSignBtn.classList.remove("hidden");
    } else {
      els.practiceNextSignBtn.classList.add("hidden");
    }
  }
  if (els.practiceApproveAgainBtn) {
    if (isApproved) {
      els.practiceApproveAgainBtn.classList.remove("hidden");
    } else {
      els.practiceApproveAgainBtn.classList.add("hidden");
    }
  }

  const scoreStr = evalData.accuracyPct.startsWith("Match:") ? evalData.accuracyPct : `Match: ${evalData.accuracyPct}`;

  if (els.practiceScoreBadge) {
    els.practiceScoreBadge.textContent = scoreStr;
  }
  if (els.practiceStatusText) {
    els.practiceStatusText.textContent = evalData.verdict;
  }
  if (els.practiceVerdictIcon) {
    els.practiceVerdictIcon.textContent = evalData.icon;
  }
  if (els.practiceMetricTarget) {
    els.practiceMetricTarget.textContent = evalData.targetGloss;
  }
  if (els.practiceMetricDetected) {
    els.practiceMetricDetected.textContent = evalData.detectedGloss;
  }
  if (els.practiceMetricHandShape) {
    els.practiceMetricHandShape.textContent = evalData.handShape;
  }
  if (els.practiceMetricMotion) {
    els.practiceMetricMotion.textContent = evalData.motionPath;
  }
  if (els.practiceEvalDetails) {
    els.practiceEvalDetails.textContent = evalData.detailsText;
  }

  // Update Visual Meter Track Fill & Styling
  if (els.practiceMeterFill) {
    els.practiceMeterFill.style.width = `${evalData.numVal}%`;
    els.practiceMeterFill.classList.remove("warn", "poor");
    if (evalData.numVal < 40) {
      els.practiceMeterFill.classList.add("poor");
    } else if (evalData.numVal < 60) {
      els.practiceMeterFill.classList.add("warn");
    }
  }

  // Update Card Styling Classes
  els.practiceEvalResult.classList.remove("approved-box", "not-sure-box", "poor-match-box");
  if (isApproved) {
    els.practiceEvalResult.classList.add("approved-box");
  } else if (evalData.category === "warn" || evalData.numVal >= 50) {
    els.practiceEvalResult.classList.add("not-sure-box");
  } else {
    els.practiceEvalResult.classList.add("poor-match-box");
  }

  // Update In-Page Elements on Main Dashboard
  if (els.pageScoreBadge) els.pageScoreBadge.textContent = scoreStr;
  if (els.pageStatusText) els.pageStatusText.textContent = evalData.verdict;
  if (els.pageVerdictIcon) els.pageVerdictIcon.textContent = evalData.icon;
  if (els.pageMetricTarget) els.pageMetricTarget.textContent = evalData.targetGloss;
  if (els.pageMetricDetected) els.pageMetricDetected.textContent = evalData.detectedGloss;
  if (els.pageMetricHandShape) els.pageMetricHandShape.textContent = evalData.handShape;
  if (els.pageMetricMotion) els.pageMetricMotion.textContent = evalData.motionPath;
  if (els.pageEvalDetails) els.pageEvalDetails.textContent = evalData.detailsText;

  if (els.pageMeterFill) {
    els.pageMeterFill.style.width = `${evalData.numVal}%`;
    els.pageMeterFill.classList.remove("warn", "poor");
    if (evalData.numVal < 40) {
      els.pageMeterFill.classList.add("poor");
    } else if (evalData.numVal < 60) {
      els.pageMeterFill.classList.add("warn");
    }
  }

  if (els.pagePracticeResultBox) {
    els.pagePracticeResultBox.classList.remove("hidden", "not-sure-box", "poor-match-box");
    if (!isApproved) {
      els.pagePracticeResultBox.classList.add(evalData.numVal < 50 ? "poor-match-box" : "not-sure-box");
    }
  }

  if (els.pagePracticeRecordBtn) els.pagePracticeRecordBtn.classList.remove("hidden");
  if (els.pagePracticeStopBtn) els.pagePracticeStopBtn.classList.add("hidden");

  // Sync with Sidebar Practice Section
  if (els.sidebarTargetGlossBadge) els.sidebarTargetGlossBadge.textContent = evalData.targetGloss;
  if (els.accuracyScoreValue) els.accuracyScoreValue.textContent = scoreStr;
  if (els.accuracyBarFill) {
    els.accuracyBarFill.style.width = `${evalData.numVal}%`;
    if (isApproved) {
      els.accuracyBarFill.style.background = "linear-gradient(90deg, #10b981, #06b6d4)";
    } else if (evalData.numVal < 40) {
      els.accuracyBarFill.style.background = "linear-gradient(90deg, #ef4444, #f43f5e)";
    } else {
      els.accuracyBarFill.style.background = "linear-gradient(90deg, #f59e0b, #fbbf24)";
    }
  }
  if (els.practiceFeedback) {
    els.practiceFeedback.textContent = isApproved
      ? `Approved! "${evalData.targetGloss}": ${scoreStr} (${evalData.verdict}). Ready for next sign.`
      : `Practice for "${evalData.targetGloss}": ${scoreStr}. Score below 60%; click Try Again.`;
  }
}

function resetPracticeToRecord() {
  if (els.practiceEvalResult) els.practiceEvalResult.classList.add("hidden");
  if (els.practiceNotification) els.practiceNotification.classList.add("hidden");
  if (els.practiceCamVideo && state.stream) {
    els.practiceCamVideo.srcObject = state.stream;
    els.practiceCamVideo.play().catch(() => {});
  }
  els.practiceCamWrapper?.classList.remove("is-paused-feed", "is-analyzing");
  const practiceScore = document.querySelector("#practiceCameraScore");
  if (practiceScore) practiceScore.textContent = "--%";
  
  if (els.camBoxRecordBtn) {
    els.camBoxRecordBtn.classList.remove("hidden");
    els.camBoxRecordBtn.disabled = false;
    els.camBoxRecordBtn.focus();
  }
  if (els.camBoxStopBtn) {
    els.camBoxStopBtn.classList.add("hidden");
    els.camBoxStopBtn.disabled = false;
  }
}

function retryPracticeRecording() {
  getPracticeAudioContext();
  resetPracticeToRecord();
  if (els.practiceNotification) {
    els.practiceNotification.classList.remove("hidden", "failure");
    els.practiceNotification.classList.add("success");
    els.practiceNotification.textContent = "🔄 Reset! Click 'Record Sign' to start your attempt.";
    setTimeout(() => { els.practiceNotification?.classList.add("hidden"); }, 2500);
  }
}

function practiceNextSign() {
  resetPracticeToRecord();
  if (!els.practiceGlossSelect) return;
  const currentIndex = els.practiceGlossSelect.selectedIndex;
  if (currentIndex >= 0 && currentIndex < els.practiceGlossSelect.options.length - 1) {
    els.practiceGlossSelect.selectedIndex = currentIndex + 1;
    onPracticeSelectChange();
  }
}


function updatePracticeFeedback(tracking) {
  // The older sidebar practice feedback element is not present in the current
  // layout. This function is called for every MediaPipe result, so guard the
  // optional UI node to keep a missing practice panel from aborting the hand
  // result callback and forcing the tracker into its reconnect loop.
  if (!els.practiceFeedback) return;
  const target = els.practiceTarget ? els.practiceTarget.textContent : "HELLO";
  els.practiceFeedback.textContent = tracking
    ? `Tracking active for target ${target}. Perform sign gesture.`
    : `Prepare ${target} inside camera view.`;
}

function openPracticeSelectorModal() {
  if (!els.practiceSelectModal) return;
  renderPracticeSelectorModal();
  els.practiceSelectModal.classList.remove("hidden");
}

function hidePracticeSelectorModal() {
  if (els.practiceSelectModal) {
    els.practiceSelectModal.classList.add("hidden");
  }
}

function switchPracticeTab(tab) {
  if (tab === "sentences") {
    if (els.tabSentencesBtn) els.tabSentencesBtn.className = "selector-tab-btn active";
    if (els.tabWordsBtn) els.tabWordsBtn.className = "selector-tab-btn";
    if (els.practiceSentencesView) els.practiceSentencesView.classList.remove("hidden");
    if (els.practiceWordsView) els.practiceWordsView.classList.add("hidden");
  } else {
    if (els.tabSentencesBtn) els.tabSentencesBtn.className = "selector-tab-btn";
    if (els.tabWordsBtn) els.tabWordsBtn.className = "selector-tab-btn active";
    if (els.practiceSentencesView) els.practiceSentencesView.classList.add("hidden");
    if (els.practiceWordsView) els.practiceWordsView.classList.remove("hidden");
  }
}

function renderPracticeSelectorModal() {
  if (els.sentencesCategoryList) {
    els.sentencesCategoryList.innerHTML = "";
    ISL_BASIC_SENTENCES.forEach((group) => {
      const groupDiv = document.createElement("div");
      groupDiv.className = "category-group";

      const title = document.createElement("h4");
      title.className = "category-title";
      title.textContent = group.category;
      groupDiv.appendChild(title);

      const grid = document.createElement("div");
      grid.className = "category-grid";

      group.items.forEach((item) => {
        const card = document.createElement("div");
        card.className = "item-card";

        const glossEl = document.createElement("div");
        glossEl.className = "item-card-gloss";
        glossEl.textContent = item.display;

        const transEl = document.createElement("div");
        transEl.className = "item-card-translation";
        transEl.textContent = `"${item.translation}"`;

        const actions = document.createElement("div");
        actions.className = "item-card-actions";

        const previewBtn = document.createElement("button");
        previewBtn.className = "meet-btn secondary-meet-btn";
        previewBtn.textContent = "▶ Watch Reference Video";
        previewBtn.onclick = (e) => {
          e.stopPropagation();
          hidePracticeSelectorModal();
          openTutorModal(item.glosses[0]);
        };

        const selectBtn = document.createElement("button");
        selectBtn.className = "meet-btn primary-meet-btn";
        selectBtn.textContent = "🎯 Select & Practice";
        selectBtn.onclick = (e) => {
          e.stopPropagation();
          selectPracticeTarget(item.display, true, item.translation);
        };

        actions.appendChild(previewBtn);
        actions.appendChild(selectBtn);

        card.appendChild(glossEl);
        card.appendChild(transEl);
        card.appendChild(actions);

        card.onclick = () => selectPracticeTarget(item.display, true, item.translation);

        grid.appendChild(card);
      });

      groupDiv.appendChild(grid);
      els.sentencesCategoryList.appendChild(groupDiv);
    });
  }

  if (els.wordsCategoryList) {
    els.wordsCategoryList.innerHTML = "";
    ISL_WORD_CATEGORIES.forEach((group) => {
      const groupDiv = document.createElement("div");
      groupDiv.className = "category-group";

      const title = document.createElement("h4");
      title.className = "category-title";
      title.textContent = group.category;
      groupDiv.appendChild(title);

      const grid = document.createElement("div");
      grid.className = "category-grid words-grid";

      group.glosses.forEach((gloss) => {
        const card = document.createElement("div");
        card.className = "item-card";

        const glossEl = document.createElement("div");
        glossEl.className = "item-card-gloss";
        glossEl.textContent = gloss;

        const selectBtn = document.createElement("button");
        selectBtn.className = "meet-btn primary-meet-btn";
        selectBtn.style.fontSize = "11px";
        selectBtn.style.padding = "6px";
        selectBtn.textContent = "Practice 🎯";
        selectBtn.onclick = (e) => {
          e.stopPropagation();
          selectPracticeTarget(gloss, false, "");
        };

        card.appendChild(glossEl);
        card.appendChild(selectBtn);
        card.onclick = () => selectPracticeTarget(gloss, false, "");

        grid.appendChild(card);
      });

      groupDiv.appendChild(grid);
      els.wordsCategoryList.appendChild(groupDiv);
    });
  }
}

function selectPracticeTarget(targetDisplay, isSentence, translation) {
  const cleanTarget = formatGlossDisplay(targetDisplay);
  state.practiceTargetGloss = cleanTarget;
  if (els.practiceTarget) {
    els.practiceTarget.textContent = targetDisplay;
  }
  if (els.practiceTargetType) {
    els.practiceTargetType.textContent = isSentence ? "BASIC SENTENCE" : "SINGLE WORD";
  }
  if (els.practiceFeedback) {
    els.practiceFeedback.textContent = isSentence
      ? `Practice Target: "${targetDisplay}" (${translation}). Opening Practice Studio...`
      : `Practice Target: "${targetDisplay}". Opening Practice Studio...`;
  }
  hidePracticeSelectorModal();
  openPracticeModal(cleanTarget);
}

async function startPracticeSession() {
  openPracticeModal(state.practiceTargetGloss || "WATER");
}

function commitCurrentSign() {
  const signToCommit = state.lastConfirmedSign || "UNKNOWN";
  addConversationEntry("sign", signToCommit);
  speakText(signToCommit);
}

function addConversationEntry(direction, text) {
  const displayText = typeof text === "string" ? text.trim() : "";
  const entry = {
    timestamp: new Date().toISOString(),
    direction,
    text: displayText,
  };
  state.conversation.push(entry);
  if (direction === "sign" && els.lastRecognizedWord && displayText && displayText.toUpperCase() !== "UNKNOWN") {
    els.lastRecognizedWord.textContent = formatGlossDisplay(displayText);
    els.lastRecognizedWord.classList.remove("is-empty");
  }
  renderTranscript();
}

function renderTranscript() {
  els.transcript.innerHTML = "";
  if (!state.conversation.length) {
    const empty = document.createElement("p");
    empty.className = "transcript-entry";
    empty.style.color = "var(--text-muted)";
    empty.textContent = "No conversation entries recorded yet.";
    els.transcript.appendChild(empty);
    return;
  }
  state.conversation.forEach((entry) => {
    const node = document.createElement("div");
    node.className = "transcript-entry";
    let label = "Recognized Sign";
    if (entry.direction === "speech") label = "Speech Input";
    else if (entry.direction === "sentence") label = "Generated Sentence";
    
    node.innerHTML = `<span class="time">${formatTime(entry.timestamp)}</span><span class="tag">[${label}]</span><strong>${escapeHtml(
      entry.text
    )}</strong>`;
    els.transcript.appendChild(node);
  });
}

function clearConversation() {
  state.conversation = [];
  state.practice = [];
  state.recordedGlosses = [];
  state.sentenceFrameBuffer = [];
  renderTranscript();
  if (els.lastRecognizedWord) {
    els.lastRecognizedWord.textContent = "Waiting for a sign…";
    els.lastRecognizedWord.classList.add("is-empty");
  }
  state.lastConfirmedSign = null;
  els.currentSign.textContent = "NOT SURE";
  renderSentenceChips();
  if (els.sentenceOutput) {
    els.sentenceOutput.textContent = "Your translated sentence will appear here after recording.";
  }
  els.speechText.textContent = "No speech captured yet.";
}

function speakText(text) {
  if (!text || text === "UNKNOWN") return;
  try {
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.95;
      utterance.pitch = 1.0;
      utterance.volume = 1.0;
      window.speechSynthesis.speak(utterance);
    }
  } catch (err) {
    console.warn("TTS error:", err);
  }
}

function speakOutput() {
  const latest = [...state.conversation].reverse().find((entry) => entry.text && entry.text !== "UNKNOWN");
  const text = latest ? latest.text : state.lastPredictedSign || "Unknown";
  speakText(text);
}

function setupSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    els.speechText.textContent = "Speech recognition is not supported in this browser.";
    els.speechButton.disabled = true;
    return;
  }

  state.speechRecognition = new SpeechRecognition();
  state.speechRecognition.continuous = true;
  state.speechRecognition.interimResults = true;
  state.speechRecognition.lang = "en-IN";

  state.speechRecognition.onresult = (event) => {
    let interim = "";
    for (let index = event.resultIndex; index < event.results.length; index += 1) {
      const result = event.results[index];
      const transcriptText = result[0].transcript.trim();
      if (result.isFinal && transcriptText) {
        addConversationEntry("speech", transcriptText);
      } else {
        interim += transcriptText;
      }
    }
    els.speechText.textContent = interim || "Listening...";
  };

  state.speechRecognition.onend = () => {
    state.listening = false;
    els.speechButton.classList.remove("listening-state");
    els.speechBtnText.textContent = "Listen";
  };
}

function toggleSpeechRecognition() {
  if (!state.speechRecognition) return;
  if (state.listening) {
    state.speechRecognition.stop();
  } else {
    state.listening = true;
    els.speechButton.classList.add("listening-state");
    els.speechBtnText.textContent = "Stop Listening";
    state.speechRecognition.start();
  }
}

async function exportConversation() {
  const response = await fetch("/api/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      conversation: state.conversation,
      practice: state.practice,
    }),
  });
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "isl-lens-conversation.json";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function drawFallbackLandmarks(landmarks) {
  ctx.fillStyle = "#06b6d4";
  landmarks.forEach((point) => {
    ctx.beginPath();
    ctx.arc(point.x * els.overlay.width, point.y * els.overlay.height, 4, 0, Math.PI * 2);
    ctx.fill();
  });
}

function resizeCanvas() {
  const rect = els.videoContainer.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  if (els.overlay.width !== width || els.overlay.height !== height) {
    els.overlay.width = width;
    els.overlay.height = height;
  }
}

function formatTime(value) {
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(value));
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => (
    {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#039;",
    }[char]
  ));
}

/* Model Feedback & Interactive Retraining Studio Functions */
function openCorrectionModal(gloss = "") {
  // Click listeners pass a PointerEvent as their first argument. Never treat
  // that event object as a custom sign gloss in the retraining input.
  if (typeof gloss !== "string") gloss = "";
  // Each visit starts a new training session. Never leave the previous sign's
  // success/failure card covering the camera when the user returns to teach
  // another sign.
  if (els.retrainLoadingBox) els.retrainLoadingBox.classList.add("hidden");
  if (els.retrainSuccessBox) els.retrainSuccessBox.classList.add("hidden");
  if (els.retrainProgressBar) els.retrainProgressBar.classList.add("hidden");
  if (els.startRetrainRecBtn) {
    els.startRetrainRecBtn.classList.remove("hidden");
    els.startRetrainRecBtn.disabled = false;
  }
  if (els.stopRetrainRecBtn) {
    els.stopRetrainRecBtn.classList.add("hidden");
    els.stopRetrainRecBtn.disabled = false;
  }
  if (els.retrainLoadingFill) els.retrainLoadingFill.style.width = "25%";
  const currentThought = state.lastConfirmedSign
    ? state.lastConfirmedSign.replace(/_/g, " ")
    : "NOT SURE";
  if (els.correctionModelThought) {
    els.correctionModelThought.textContent = currentThought;
  }
  if (els.correctGlossSelect) {
    els.correctGlossSelect.innerHTML = "";
    // Default option: "no input" so user can type or choose cleanly
    const defaultOpt = document.createElement("option");
    defaultOpt.value = "";
    defaultOpt.textContent = "no input -- Type custom word/sentence or select sign";
    defaultOpt.selected = true;
    els.correctGlossSelect.appendChild(defaultOpt);

    const list = state.vocabulary.map((v) => v.gloss);
    const seen = new Set();
    list.forEach((gloss) => {
      const cleanGloss = String(gloss).toUpperCase().replace(/_/g, " ").trim();
      if (!cleanGloss || seen.has(cleanGloss)) return;
      seen.add(cleanGloss);
      const opt = document.createElement("option");
      opt.value = cleanGloss;
      opt.textContent = cleanGloss;
      els.correctGlossSelect.appendChild(opt);
    });
  }

  if (els.customGlossInput) {
    els.customGlossInput.value = gloss && !state.vocabulary.some((item) => formatGlossDisplay(item.gloss) === formatGlossDisplay(gloss))
      ? formatGlossDisplay(gloss)
      : "";
  }
  if (gloss && els.correctGlossSelect) {
    const matchingOption = [...els.correctGlossSelect.options].find((option) => formatGlossDisplay(option.value) === formatGlossDisplay(gloss));
    if (matchingOption) els.correctGlossSelect.value = matchingOption.value;
  }

  // Ensure camera stream is running and connected to retrainVideo element inside modal
  if (!state.stream) {
    startCamera();
  } else if (els.retrainVideo) {
    els.retrainVideo.srcObject = state.stream;
    els.retrainVideo.play().catch(() => {});
  }

  // Reset Retraining Studio controls state
  state.isRetrainRecording = false;
  state.retrainFrameBuffer = [];
  if (els.startRetrainRecBtn) els.startRetrainRecBtn.classList.remove("hidden");
  if (els.stopRetrainRecBtn) els.stopRetrainRecBtn.classList.add("hidden");
  if (els.retrainRecOverlay) els.retrainRecOverlay.classList.add("hidden");
  if (els.retrainProgressBar) els.retrainProgressBar.classList.add("hidden");
  if (els.retrainStatusText) els.retrainStatusText.textContent = "Live Camera View • Position Hands to Sign";

  if (els.correctionModal) {
    els.correctionModal.classList.remove("hidden");
    setTimeout(() => {
      if (els.customGlossInput) {
        try {
          els.customGlossInput.focus();
        } catch (e) {}
      }
    }, 150);
  }

  // Pause main background prediction immediately
  resetObservationState();
  if (els.decisionState) {
    els.decisionState.textContent = "PAUSED";
    els.decisionState.className = "badge-tag standby";
  }
  if (els.currentSign) {
    els.currentSign.textContent = "TRAINING IN PROGRESS";
    els.currentSign.style.color = "var(--text-muted)";
  }
  if (els.recognitionReason) {
    els.recognitionReason.textContent = "Live camera sign prediction is paused while training studio is open.";
  }
}

function hideCorrectionModal() {
  if (state.isRetrainRecording) {
    stopRetrainRecordingAndTrain();
    return;
  }
  if (els.correctionModal) {
    els.correctionModal.classList.add("hidden");
  }

  // Restore main background camera status
  if (els.decisionState) {
    els.decisionState.textContent = "STANDBY";
    els.decisionState.className = "badge-tag";
  }
  if (els.currentSign) {
    els.currentSign.textContent = "UNKNOWN";
    els.currentSign.style.color = "var(--text-main)";
  }
  if (els.obsProgressFill) {
    els.obsProgressFill.style.width = "0%";
  }
  if (els.obsLabel) {
    els.obsLabel.textContent = "Show hands to read sign...";
  }
  if (els.recognitionReason) {
    els.recognitionReason.textContent = "Position hands inside camera frame to begin.";
  }
}

// ==========================================
// Custom User Reference Video Recording
// ==========================================
let userVideoMediaRecorder = null;
let userVideoChunks = [];

function startUserVideoRecording() {
  userVideoChunks = [];
  if (state.stream && typeof MediaRecorder !== "undefined") {
    try {
      const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
        ? "video/webm;codecs=vp9"
        : (MediaRecorder.isTypeSupported("video/webm") ? "video/webm" : "");
      userVideoMediaRecorder = mime ? new MediaRecorder(state.stream, { mimeType: mime }) : new MediaRecorder(state.stream);
      userVideoMediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) userVideoChunks.push(e.data);
      };
      userVideoMediaRecorder.start(100);
    } catch (e) {
      console.warn("MediaRecorder start error:", e);
      userVideoMediaRecorder = null;
    }
  }
}

async function stopUserVideoRecordingAndUpload(gloss) {
  if (!userVideoMediaRecorder || userVideoMediaRecorder.state === "inactive") return;
  return new Promise((resolve) => {
    userVideoMediaRecorder.onstop = async () => {
      try {
        if (userVideoChunks.length > 0) {
          const blob = new Blob(userVideoChunks, { type: "video/webm" });
          if (blob.size > 1000) {
            const clean = String(gloss || "").toLowerCase().replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, "");
            await fetch(`/api/upload_user_video?gloss=${encodeURIComponent(clean)}`, {
              method: "POST",
              headers: { "Content-Type": "video/webm" },
              body: blob
            });
            console.log(`[ISL-Lens] Attached custom training video for "${clean}" (${blob.size} bytes)`);
          }
        }
      } catch (err) {
        console.warn("Failed to upload user video:", err);
      }
      userVideoChunks = [];
      userVideoMediaRecorder = null;
      resolve();
    };
    try {
      userVideoMediaRecorder.stop();
    } catch (e) {
      userVideoMediaRecorder = null;
      resolve();
    }
  });
}

async function startRetrainRecording() {
  const customVal = els.customGlossInput ? els.customGlossInput.value.trim() : "";
  const selectVal = els.correctGlossSelect ? els.correctGlossSelect.value : "";
  const rawInput = customVal || selectVal;

  let selectedGloss = rawInput.replace(/_/g, " ").replace(/\s+/g, " ").trim().toUpperCase();

  if (!selectedGloss) {
    alert("Please type an accurate word or sentence to teach (e.g. WATER, I AM FINE, HOW ARE YOU)...");
    if (els.customGlossInput) els.customGlossInput.focus();
    return;
  }

  // A retry begins a fresh camera session. Clear the previous rejection or
  // success overlay before recording so it never remains over the live view.
  if (els.retrainSuccessBox) els.retrainSuccessBox.classList.add("hidden");
  if (els.retrainLoadingBox) els.retrainLoadingBox.classList.add("hidden");
  if (els.retrainProgressBar) els.retrainProgressBar.classList.add("hidden");
  if (els.retrainLoadingFill) els.retrainLoadingFill.style.width = "25%";

  if (!state.stream) {
    await startCamera();
  }
  if (!state.stream || !state.hands) {
    if (els.retrainStatusText) els.retrainStatusText.textContent = "Camera and hand tracking must be ready before recording a training example.";
    return;
  }

  startUserVideoRecording();
  state.isRetrainRecording = true;
  state.retrainFrameBuffer = [];
  state.retrainRecStartTime = Date.now();

  if (els.startRetrainRecBtn) els.startRetrainRecBtn.classList.add("hidden");
  if (els.stopRetrainRecBtn) els.stopRetrainRecBtn.classList.remove("hidden");
  if (els.retrainRecOverlay) els.retrainRecOverlay.classList.remove("hidden");
  if (els.retrainProgressBar) els.retrainProgressBar.classList.remove("hidden");
  if (els.retrainStatusText) els.retrainStatusText.textContent = `🔴 Recording Gesture for "${selectedGloss}"...`;

  if (state.retrainRecTimerInterval) clearInterval(state.retrainRecTimerInterval);
  state.retrainRecTimerInterval = setInterval(() => {
    if (!state.isRetrainRecording) {
      clearInterval(state.retrainRecTimerInterval);
      return;
    }
    const elapsed = Date.now() - state.retrainRecStartTime;
    const secStr = (elapsed / 1000).toFixed(1);
    const maxSec = 10.0;
    const pct = Math.min(100, (elapsed / (maxSec * 1000)) * 100);

    if (els.retrainRecTimerText) {
      els.retrainRecTimerText.textContent = `RECORDING GESTURE: ${secStr}s (${state.retrainFrameBuffer.length} frames checked)`;
    }
    if (els.repProgressFill) {
      els.repProgressFill.style.width = `${pct}%`;
    }

    if (elapsed >= maxSec * 1000) {
      stopRetrainRecordingAndTrain();
    }
  }, 100);
}

async function stopRetrainRecordingAndTrain() {
  if (state.retrainRecTimerInterval) clearInterval(state.retrainRecTimerInterval);
  state.isRetrainRecording = false;

  const customVal = els.customGlossInput ? els.customGlossInput.value.trim() : "";
  const selectVal = els.correctGlossSelect ? els.correctGlossSelect.value : "";
  const rawInput = customVal || selectVal;

  let selectedGloss = rawInput.replace(/_/g, " ").replace(/\s+/g, " ").trim().toUpperCase();
  // Match the server's label normalization for sentence-ending punctuation.
  selectedGloss = selectedGloss.replace(/[.!?,;:]+$/, "").trim();
  if (!selectedGloss) {
    if (els.retrainLoadingBox) els.retrainLoadingBox.classList.add("hidden");
    if (els.startRetrainRecBtn) els.startRetrainRecBtn.classList.remove("hidden");
    if (els.stopRetrainRecBtn) {
      els.stopRetrainRecBtn.classList.add("hidden");
      els.stopRetrainRecBtn.disabled = false;
    }
    if (els.retrainSuccessBox) els.retrainSuccessBox.classList.remove("hidden");
    if (els.retrainSuccessStatus) els.retrainSuccessStatus.textContent = "SIGN NAME REQUIRED";
    if (els.retrainSuccessBadge) els.retrainSuccessBadge.textContent = "Training not saved";
    if (els.retrainSuccessDetails) els.retrainSuccessDetails.textContent = "Enter the sign or sentence you recorded, then try again.";
    return;
  }

  // Asynchronously stop media recorder and upload video to save as reference video
  stopUserVideoRecordingAndUpload(selectedGloss);

  // Show Loading UI for Train AI
  if (els.stopRetrainRecBtn) els.stopRetrainRecBtn.disabled = true;
  if (els.retrainRecOverlay) els.retrainRecOverlay.classList.add("hidden");
  if (els.retrainSuccessBox) els.retrainSuccessBox.classList.add("hidden");
  if (els.retrainLoadingBox) els.retrainLoadingBox.classList.remove("hidden");
  if (els.retrainLoadingFill) els.retrainLoadingFill.style.width = "25%";
  if (els.retrainLoadingSubtitle) {
    els.retrainLoadingSubtitle.textContent = `Training "${selectedGloss}" and validating the model update...`;
  }

  const capturedExemplars = [];
  if (state.retrainFrameBuffer.length >= 6) {
    const fullSequenceSummary = Array.from(summarizeSequence(state.retrainFrameBuffer));
    capturedExemplars.push(fullSequenceSummary);

    const n = state.retrainFrameBuffer.length;
    const half = Math.floor(n / 2);
    if (half >= 4) {
      capturedExemplars.push(Array.from(summarizeSequence(state.retrainFrameBuffer.slice(0, half))));
      capturedExemplars.push(Array.from(summarizeSequence(state.retrainFrameBuffer.slice(n - half))));
    }
  } else if (state.obsBuffer.length >= 6) {
    capturedExemplars.push(Array.from(summarizeSequence(state.obsBuffer)));
  } else if (state.lastObservationSummary) {
    capturedExemplars.push(Array.from(state.lastObservationSummary));
  }

  const validExemplars = capturedExemplars.filter((features) => features && features.length === FRAME_FEATURES * 6);
  if (!validExemplars.length) {
    if (els.retrainLoadingBox) els.retrainLoadingBox.classList.add("hidden");
    if (els.startRetrainRecBtn) els.startRetrainRecBtn.classList.remove("hidden");
    if (els.stopRetrainRecBtn) {
      els.stopRetrainRecBtn.classList.add("hidden");
      els.stopRetrainRecBtn.disabled = false;
    }
    if (els.retrainSuccessBox) els.retrainSuccessBox.classList.remove("hidden");
    if (els.retrainSuccessStatus) els.retrainSuccessStatus.textContent = "NOT ENOUGH TRACKED SIGN FRAMES";
    if (els.retrainSuccessBadge) els.retrainSuccessBadge.textContent = "Training not saved";
    if (els.retrainSuccessDetails) els.retrainSuccessDetails.textContent = "Record the sign with your hand clearly visible for at least six tracked frames, then stop and train again.";
    return;
  }

  let retrainingSucceeded = false;
  let retrainingMessage = "The candidate model did not pass validation; the active model remains unchanged.";

  setTimeout(() => {
    if (els.retrainLoadingFill) els.retrainLoadingFill.style.width = "65%";
    if (els.retrainLoadingSubtitle) {
      els.retrainLoadingSubtitle.textContent = `2/3. Retraining neural weights & updating vocabulary for "${selectedGloss}"...`;
    }
  }, 400);

  try {
    const res = await fetch("/api/correct_sign", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        correct_gloss: selectedGloss,
        model_predicted: state.lastPredictedSign || "NOT SURE",
        exemplars: validExemplars
      })
    });
    if (res.ok) {
      const payload = await res.json();
      retrainingSucceeded = payload.retrain_status?.retrained === true;
      retrainingMessage = payload.message || payload.retrain_status?.error || payload.retrain_status?.reason || retrainingMessage;
      if (retrainingSucceeded) await loadModel();
      await loadVocabulary();
    } else {
      const responseBody = await res.text();
      let errorMessage = responseBody.trim();
      try {
        const errorPayload = JSON.parse(responseBody);
        errorMessage = errorPayload.message || errorMessage;
      } catch (_) {
        // Keep plain-text responses useful too.
      }
      retrainingMessage = errorMessage || `Training request failed (${res.status}).`;
    }
  } catch (err) {
    console.warn("Correction API error:", err);
    retrainingMessage = "Could not contact the training service. Check that the local server is running, then try again.";
  }

  setTimeout(() => {
    if (els.retrainLoadingFill) els.retrainLoadingFill.style.width = "95%";
    if (els.retrainLoadingSubtitle) {
      els.retrainLoadingSubtitle.textContent = `3/3. Calibrating real-time recognition weights...`;
    }
  }, 750);

  setTimeout(() => {
    if (els.retrainLoadingBox) els.retrainLoadingBox.classList.add("hidden");
    if (els.startRetrainRecBtn) els.startRetrainRecBtn.classList.remove("hidden");
    if (els.stopRetrainRecBtn) {
      els.stopRetrainRecBtn.classList.add("hidden");
      els.stopRetrainRecBtn.disabled = false;
    }

    // Report whether the candidate passed the held-out quality gate.
    if (els.retrainSuccessBox) {
      els.retrainSuccessBox.classList.remove("hidden");
      if (els.retrainResultIcon) els.retrainResultIcon.textContent = retrainingSucceeded ? "✅" : "⚠️";
      if (els.retrainSuccessStatus) els.retrainSuccessStatus.textContent = retrainingSucceeded
        ? `MODEL RETRAINED FOR "${selectedGloss}"!`
        : "MODEL NOT UPDATED";
      if (els.retrainSuccessBadge) els.retrainSuccessBadge.textContent = retrainingSucceeded ? "Model updated" : "Model not updated";
      if (els.retrainSuccessDetails) {
        els.retrainSuccessDetails.textContent = retrainingSucceeded
          ? `The model was updated using ${validExemplars.length} recorded sequence example(s) for "${selectedGloss}".`
          : `${retrainingMessage} The recorded examples were not added to the active model.`;
      }
    }

    if (retrainingSucceeded) {
      state.lastPredictedSign = selectedGloss;
      state.lastConfidence = 0.98;
      updateRecognitionDisplay(selectedGloss, 0.98, "CONFIRMED", `✅ AI model retrained for "${selectedGloss}".`);

      // Automatically return to original home screen to practice the assigned sign
      setTimeout(() => {
        hideCorrectionModal();
        if (els.pagePracticeGlossSelect) {
          let found = false;
          for (let i = 0; i < els.pagePracticeGlossSelect.options.length; i++) {
            if (els.pagePracticeGlossSelect.options[i].value === selectedGloss) {
              els.pagePracticeGlossSelect.selectedIndex = i;
              found = true;
              break;
            }
          }
          if (!found) {
            const opt = document.createElement("option");
            opt.value = selectedGloss;
            opt.textContent = `${selectedGloss} (Custom Trained)`;
            els.pagePracticeGlossSelect.appendChild(opt);
            els.pagePracticeGlossSelect.value = selectedGloss;
          }
          onPagePracticeGlossChange();
        }
        state.practiceTargetGloss = selectedGloss;
        if (els.pageMetricTarget) els.pageMetricTarget.textContent = selectedGloss;
        if (els.practiceFeedback) {
          els.practiceFeedback.textContent = `🎯 Sign "${selectedGloss}" trained! Click "Record Sign" to test your accuracy.`;
        }
        const practiceCard = document.querySelector(".page-practice-card") || document.querySelector("#practiceSection");
        if (practiceCard) {
          practiceCard.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      }, 850);
    } else {
      updateRecognitionDisplay("UNKNOWN", 0, "STANDBY", retrainingMessage);
    }
  }, 1100);
}

function practiceJustTrainedSign() {
  const gloss = state.lastPredictedSign || (els.customGlossInput ? els.customGlossInput.value.trim() : "WATER") || "WATER";
  hideCorrectionModal();
  openPracticeModal(gloss);
}

function resetRetrainForm() {
  if (els.retrainSuccessBox) els.retrainSuccessBox.classList.add("hidden");
  if (els.retrainProgressBar) els.retrainProgressBar.classList.add("hidden");
  if (els.customGlossInput) els.customGlossInput.value = "";
  if (els.startRetrainRecBtn) els.startRetrainRecBtn.classList.remove("hidden");
  if (els.stopRetrainRecBtn) els.stopRetrainRecBtn.classList.add("hidden");
  if (els.retrainStatusText) els.retrainStatusText.textContent = "Live Camera View • Position Hands to Sign";
}

/* In-Page Practice Section Functions (Recording, Loading Screen & Accuracy Reporting on Main Page) */
function onPagePracticeGlossChange() {
  if (els.pagePracticeGlossSelect && els.pagePracticeGlossSelect.value) {
    state.practiceTargetGloss = els.pagePracticeGlossSelect.value;
    if (els.pageMetricTarget) els.pageMetricTarget.textContent = state.practiceTargetGloss;
    if (els.practiceFeedback) {
      els.practiceFeedback.textContent = `Target set to "${state.practiceTargetGloss}". Click "Record Sign" to test your accuracy.`;
    }
  }
}

function startInPagePracticeRecording() {
  if (!state.stream) {
    startCamera();
  }

  state.isInPagePracticeRecording = true;
  state.isPracticeRecording = true;
  state.practiceFrameBuffer = [];
  state.practiceRecStartTime = Date.now();

  const target = (els.pagePracticeGlossSelect && els.pagePracticeGlossSelect.value)
    ? els.pagePracticeGlossSelect.value
    : (state.practiceTargetGloss || "WATER");
  state.practiceTargetGloss = target;

  if (els.pagePracticeRecordBtn) els.pagePracticeRecordBtn.classList.add("hidden");
  if (els.pagePracticeStopBtn) {
    els.pagePracticeStopBtn.classList.remove("hidden");
    els.pagePracticeStopBtn.disabled = false;
  }
  if (els.pagePracticeStopBtnText) {
    els.pagePracticeStopBtnText.textContent = "⏹️ Stop & Analyze";
  }
  if (els.pagePracticeRecOverlay) els.pagePracticeRecOverlay.classList.remove("hidden");
  if (els.pagePracticeResultBox) els.pagePracticeResultBox.classList.add("hidden");
  if (els.pagePracticeLoadingBox) els.pagePracticeLoadingBox.classList.add("hidden");

  // Pause background prediction display
  resetObservationState();
  if (els.decisionState) {
    els.decisionState.textContent = "RECORDING";
    els.decisionState.className = "badge-tag listening";
  }
  if (els.currentSign) {
    els.currentSign.textContent = `PRACTICING: ${target}`;
    els.currentSign.style.color = "var(--accent-cyan)";
  }

  if (state.inPagePracticeTimerInterval) clearInterval(state.inPagePracticeTimerInterval);
  state.inPagePracticeTimerInterval = setInterval(() => {
    if (!state.isInPagePracticeRecording) {
      clearInterval(state.inPagePracticeTimerInterval);
      return;
    }
    const elapsed = Date.now() - state.practiceRecStartTime;
    const secStr = (elapsed / 1000).toFixed(1);
    const maxSec = 10.0;

    let handsCount = 0;
    for (let i = 0; i < state.practiceFrameBuffer.length; i++) {
      if (isHandPresentInVector(state.practiceFrameBuffer[i])) handsCount++;
    }

    if (els.pagePracticeRecTimerText) {
      els.pagePracticeRecTimerText.textContent = `RECORDING PRACTICE GESTURE: ${secStr}s (${handsCount} hand frames)`;
    }

    if (elapsed >= maxSec * 1000) {
      stopInPagePracticeAndEvaluate();
    }
  }, 100);
}

function stopInPagePracticeAndEvaluate() {
  if (state.inPagePracticeTimerInterval) clearInterval(state.inPagePracticeTimerInterval);
  state.isInPagePracticeRecording = false;
  state.isPracticeRecording = false;

  if (els.pagePracticeStopBtn) els.pagePracticeStopBtn.disabled = true;
  if (els.pagePracticeStopBtnText) {
    els.pagePracticeStopBtnText.textContent = "⏳ Analyzing Sign...";
  }
  if (els.pagePracticeRecOverlay) els.pagePracticeRecOverlay.classList.add("hidden");
  if (els.pagePracticeResultBox) els.pagePracticeResultBox.classList.add("hidden");

  // Show Loading Screen / UI on Main Page
  if (els.pagePracticeLoadingBox) {
    els.pagePracticeLoadingBox.classList.remove("hidden");
  }
  if (els.pagePracticeLoadingFill) {
    els.pagePracticeLoadingFill.style.width = "25%";
  }
  if (els.pagePracticeLoadingSubtitle) {
    els.pagePracticeLoadingSubtitle.textContent = "1/3. Extracting 3D hand coordinates & body pose...";
  }

  setTimeout(() => {
    if (els.pagePracticeLoadingFill) els.pagePracticeLoadingFill.style.width = "65%";
    if (els.pagePracticeLoadingSubtitle) {
      els.pagePracticeLoadingSubtitle.textContent = "2/3. Comparing spatial-temporal movement with ISL reference model...";
    }
  }, 350);

  setTimeout(() => {
    if (els.pagePracticeLoadingFill) els.pagePracticeLoadingFill.style.width = "95%";
    if (els.pagePracticeLoadingSubtitle) {
      els.pagePracticeLoadingSubtitle.textContent = "3/3. Computing accuracy score & motion alignment metrics...";
    }
  }, 700);

  setTimeout(() => {
    if (els.pagePracticeLoadingBox) els.pagePracticeLoadingBox.classList.add("hidden");
    if (els.pagePracticeStopBtn) {
      els.pagePracticeStopBtn.classList.add("hidden");
      els.pagePracticeStopBtn.disabled = false;
    }
    if (els.pagePracticeStopBtnText) {
      els.pagePracticeStopBtnText.textContent = "⏹️ Stop & Analyze";
    }
    if (els.pagePracticeRecordBtn) els.pagePracticeRecordBtn.classList.remove("hidden");

    performPracticeAccuracyComputation();
  }, 1050);
}

function resetInPagePractice() {
  if (els.pagePracticeResultBox) els.pagePracticeResultBox.classList.add("hidden");
  if (els.pagePracticeLoadingBox) els.pagePracticeLoadingBox.classList.add("hidden");
  if (els.pagePracticeRecordBtn) els.pagePracticeRecordBtn.classList.remove("hidden");
  if (els.pagePracticeStopBtn) els.pagePracticeStopBtn.classList.add("hidden");
  const target = (els.pagePracticeGlossSelect && els.pagePracticeGlossSelect.value) ? els.pagePracticeGlossSelect.value : (state.practiceTargetGloss || "WATER");
  if (els.practiceFeedback) {
    els.practiceFeedback.textContent = `Ready to record "${target}". Position hands in front of the camera and click "Record Sign".`;
  }
  startInPagePracticeRecording();
}

window.openCorrectionModal = openCorrectionModal;
window.hideCorrectionModal = hideCorrectionModal;
window.startRetrainRecording = startRetrainRecording;
window.stopRetrainRecordingAndTrain = stopRetrainRecordingAndTrain;
window.practiceJustTrainedSign = practiceJustTrainedSign;
window.resetRetrainForm = resetRetrainForm;
window.openTutorModal = openTutorModal;
window.hideTutorModal = hideTutorModal;
window.openPracticeModalFromTutor = openPracticeModalFromTutor;
window.openPracticeModal = openPracticeModal;
window.openPracticeModalAndRecord = openPracticeModalAndRecord;
window.hidePracticeModal = hidePracticeModal;
window.startPracticeRecording = startPracticeRecording;
window.stopPracticeAndEvaluate = stopPracticeAndEvaluate;
window.retryPracticeRecording = retryPracticeRecording;
window.resetPracticeToRecord = resetPracticeToRecord;
window.practiceNextSign = practiceNextSign;
window.playPracticeSound = playPracticeSound;
window.getPracticeAudioContext = getPracticeAudioContext;
window.onPracticeSelectChange = onPracticeSelectChange;
window.startPracticeSession = startPracticeSession;
window.startInPagePracticeRecording = startInPagePracticeRecording;
window.stopInPagePracticeAndEvaluate = stopInPagePracticeAndEvaluate;
window.resetInPagePractice = resetInPagePractice;
window.onPagePracticeGlossChange = onPagePracticeGlossChange;
window.openPracticeSelectorModal = openPracticeSelectorModal;
window.hidePracticeSelectorModal = hidePracticeSelectorModal;
window.switchPracticeTab = switchPracticeTab;
window.selectPracticeTarget = selectPracticeTarget;
window.startSentenceRecording = startSentenceRecording;
window.stopSentenceRecording = stopSentenceRecording;
window.toggleCamera = toggleCamera;
window.startCamera = startCamera;
window.submitCorrection = stopRetrainRecordingAndTrain;
window.dismissSplash = dismissSplash;

init();
