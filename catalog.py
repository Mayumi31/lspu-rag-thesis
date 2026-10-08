"""
catalog.py — single source of truth for colleges and programs.

Used by:
  - templates/explore.html          (college cards + campus map)
  - templates/college-details.html  (full program details per college)
  - templates/others.html           (recommendation interest checkboxes, via JS)

To add or edit a college or program, change it HERE only.

IMAGES
  Each program has an "image" filename. Put the real photo in
  assets/images/programs/<that filename>. Until the file exists, the page
  automatically shows assets/images/programs/placeholder.svg instead.

VERIFICATION
  "desc", "study", and the "more" blocks (field overview, activities,
  skills, careers) are prototype/illustrative content, not confirmed
  LSPU-LB curriculum or outcomes. Pages label them that way. Replace with
  verified info before using this outside the thesis defense.
"""

COLLEGES = [
    {
        "id": "fisheries",
        "name": "College of Fisheries",
        "kicker": "Flagship program of LSPU-LB",
        "blurb": "Study aquatic life and sustainable fishery systems, from Laguna de Bay to the open sea.",
        "interest": "Aquatic life, the ocean, fish farming, or the environment",
        "recommend": "Explore aquatic life, fisheries, and sustainable resource management.",
        "accent": "#0b62c8", "tint": "#e9f2ff", "program_accent": "#0b62c8",
        "icon": '<path d="M6.5 12c3-4 8-6 13-4-1 2-1 6 0 8-5 2-10 0-13-4Z"/><circle cx="9.5" cy="10.5" r=".5" fill="currentColor"/><path d="M2 12c1.2-1.3 2.7-1.3 4.5 0-1.8 1.3-3.3 1.3-4.5 0Z"/>',
        "programs": [
            {"code": "BSF", "name": "BS Fisheries", "image": "bs-fisheries.jpg",
             "desc": "Aquaculture, fisheries biology, and sustainable resource management.",
             "study": ["Aquaculture", "Fisheries biology", "Sustainable resource management"]},
        ],
        "more": {
            "overview": "Fisheries is the flagship specialization of LSPU-Los Baños. The program studies aquaculture, fish biology, and the sustainable management of aquatic resources, right at the doorstep of Laguna de Bay.",
            "activities": ["Run hands-on fish and shellfish culture projects", "Study water quality and aquatic ecosystems", "Visit hatcheries, fish ponds, and coastal communities", "Design sustainable aquaculture systems"],
            "skills": ["Field and laboratory observation", "Environmental awareness", "Patience and precision"],
            "careers": ["Fisheries Technologist", "Aquaculture Farm Manager", "Fisheries Extension Officer"],
        },
    },
    {
        "id": "food",
        "name": "College of Food, Nutrition and Dietetics",
        "kicker": "Science of food and wellbeing",
        "blurb": "Turn an interest in food and health into a career planning diets, products, and community nutrition.",
        "interest": "Food, cooking science, health, or nutrition",
        "recommend": "Explore nutrition, community health, food science, and product development.",
        "accent": "#2f9e44", "tint": "#eaf7ec", "program_accent": "#2f9e44",
        "icon": '<path d="M12 22V10"/><path d="M12 10c0-4 3-7 6-7 0 4-2 7-6 7Z"/><path d="M12 14c0-4-3-7-6-7 0 4 2 7 6 7Z"/>',
        "programs": [
            {"code": "ND", "name": "BS Nutrition and Dietetics", "image": "bs-nutrition-dietetics.jpg",
             "desc": "Clinical nutrition, diet planning, and community health programs.",
             "study": ["Clinical nutrition", "Diet planning", "Community health programs"]},
            {"code": "FT", "name": "BS Food Technology", "image": "bs-food-technology.jpg",
             "desc": "Food science, processing, and product development.",
             "study": ["Food science", "Food processing", "Product development"]},
        ],
        "more": {
            "overview": "This college studies the science of food, from nutrient composition to meal planning, and prepares students to guide the health of individuals and communities.",
            "activities": ["Analyze the nutritional value of local dishes", "Plan diets for different age groups and conditions", "Practice food product development in the lab kitchen", "Run community feeding and nutrition programs"],
            "skills": ["Scientific reasoning", "Care for people's wellbeing", "Attention to detail"],
            "careers": ["Registered Nutritionist-Dietitian", "Food Technologist", "Community Health Educator"],
        },
    },
    {
        "id": "computer",
        "name": "College of Computer Studies",
        "kicker": "Build with logic and code",
        "blurb": "Build solutions through technology, programming, and innovation.",
        "interest": "Computers, coding, apps, or building things with technology",
        "recommend": "Explore computing, software, technology, and digital problem-solving.",
        "accent": "#c9a600", "tint": "#fbf6df", "program_accent": "#a68a00",
        "icon": '<rect x="7" y="7" width="10" height="10" rx="1"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3M5 5l2 2M17 17l2 2M19 5l-2 2M7 17l-2 2"/>',
        "programs": [
            {"code": "CS", "name": "BS Computer Science", "image": "bs-computer-science.jpg",
             "desc": "Explore computing, software development, intelligent systems, and related areas.",
             "study": ["Computing fundamentals", "Software development", "Intelligent systems"]},
            {"code": "IT", "name": "BS Information Technology", "image": "bs-information-technology.jpg",
             "desc": "Learn about information technology, applications, services, and specialized areas.",
             "study": ["Applications", "IT services", "Specialized IT areas"]},
        ],
        "more": {
            "overview": "Computer Studies focuses on algorithms, software development, and solving real problems through technology, from mobile apps to community information systems.",
            "activities": ["Create applications and websites", "Build and maintain information systems", "Learn programming languages and frameworks", "Work on team capstone projects"],
            "skills": ["Logical thinking", "Problem solving", "Creativity"],
            "careers": ["Software Developer", "Data Analyst", "Systems Administrator"],
        },
    },
    {
        "id": "criminal-justice",
        "name": "College of Criminal Justice Education",
        "kicker": "Serve and protect the community",
        "blurb": "Prepare for a career in law enforcement, justice, and public safety.",
        "interest": "Law, justice, investigation, or public safety",
        "recommend": "Explore public safety, criminal justice, and community service.",
        "accent": "#c92a2a", "tint": "#fbeaea", "program_accent": "#c92a2a",
        "icon": '<path d="M12 3 4 6v6c0 4.5 3 7.7 8 9 5-1.3 8-4.5 8-9V6l-8-3Z"/><path d="m9 12 2 2 4-4"/>',
        "programs": [
            {"code": "CR", "name": "BS Criminology", "image": "bs-criminology.jpg",
             "desc": "Law enforcement, criminal investigation, and public safety administration.",
             "study": ["Law enforcement", "Criminal investigation", "Public safety administration"]},
        ],
        "more": {
            "overview": "This college trains future criminologists in criminal law, investigation, and forensics, preparing students for the Criminologist Licensure Examination and careers in public safety.",
            "activities": ["Study criminal law and criminology theory", "Practice crime scene investigation techniques", "Train in physical fitness and defensive tactics", "Join community outreach and peacekeeping activities"],
            "skills": ["Discipline and integrity", "Quick, sound judgment", "Physical and mental resilience"],
            "careers": ["Criminologist", "Law Enforcement Officer", "Forensic Investigator"],
        },
    },
    {
        "id": "business",
        "name": "College of Business Administration",
        "kicker": "Lead ventures and organizations",
        "blurb": "Learn to manage, market, and grow organizations and ventures of every size.",
        "interest": "Business, marketing, money, or starting a venture",
        "recommend": "Explore organizations, marketing, finance, and entrepreneurship.",
        "accent": "#0f9aa4", "tint": "#e5f6f7", "program_accent": "#0f9aa4",
        "icon": '<rect x="3" y="8" width="18" height="12" rx="1.5"/><path d="M8 8V6a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><path d="M3 13h18"/>',
        "programs": [
            {"code": "MM", "name": "BSBA — Marketing Management", "image": "bsba-marketing-management.jpg",
             "desc": "Brand strategy, market research, and consumer behavior.",
             "study": ["Brand strategy", "Market research", "Consumer behavior"]},
            {"code": "FM", "name": "BSBA — Financial Management", "image": "bsba-financial-management.jpg",
             "desc": "Corporate finance, investment analysis, and financial planning.",
             "study": ["Corporate finance", "Investment analysis", "Financial planning"]},
        ],
        "more": {
            "overview": "Business Administration covers management, marketing, and finance, preparing students to run organizations, launch ventures, and lead teams.",
            "activities": ["Pitch and plan small business ventures", "Study marketing, finance, and operations", "Join case competitions and student organization leadership", "Complete on-the-job training in a real company"],
            "skills": ["Persuasion and negotiation", "Numerical reasoning", "Organization"],
            "careers": ["Entrepreneur", "Marketing Officer", "Financial Analyst"],
        },
    },
    {
        "id": "teacher",
        "name": "College of Teacher Education",
        "kicker": "Shape the next generation",
        "blurb": "Learn to teach, mentor, and design learning experiences for young minds.",
        "interest": "Teaching, mentoring, or helping others learn",
        "recommend": "Explore teaching, mentoring, and learning across grade levels.",
        "accent": "#7c3aed", "tint": "#f1ebfd", "program_accent": "#7c3aed",
        "icon": '<path d="M12 6c-2-1.5-5-2-8-1v13c3-1 6-.5 8 1 2-1.5 5-2 8-1V5c-3-1-6-.5-8 1Z"/><path d="M12 6v13"/>',
        "programs": [
            {"code": "EE", "name": "Bachelor of Elementary Education", "image": "bachelor-elementary-education.jpg",
             "desc": "Foundational pedagogy for teaching young learners.",
             "study": ["Foundational pedagogy", "Teaching young learners"]},
            {"code": "SE", "name": "Bachelor of Secondary Education", "image": "bachelor-secondary-education.jpg",
             "desc": "Subject-focused teaching methods for junior and senior high school.",
             "study": ["Subject-focused teaching methods", "Junior and senior high school teaching"]},
        ],
        "more": {
            "overview": "Teacher Education prepares students to plan lessons, manage classrooms, and mentor learners, building a foundation for shaping the next generation.",
            "activities": ["Design lesson plans and learning activities", "Practice teaching in real classrooms", "Study child and adolescent development", "Create instructional materials"],
            "skills": ["Clear communication", "Patience", "Creativity in explaining ideas"],
            "careers": ["Elementary or Secondary Teacher", "Curriculum Developer", "Guidance Associate"],
        },
    },
    {
        "id": "tourism-hospitality",
        "name": "College of Tourism and Hospitality Management",
        "kicker": "Craft memorable experience",
        "blurb": "Design travel experiences and run hotels, restaurants, and events with heart.",
        "interest": "Travel, hospitality, and event experiences",
        "recommend": "Explore travel experiences, hospitality, hotels, restaurants, and event services.",
        "accent": "#c47700", "tint": "#fff3df", "program_accent": "#c47700",
        "icon": '<circle cx="12" cy="12" r="9.5"/><path d="m15.8 8.2-2.6 5-5 2.6 2.6-5 5-2.6Z"/><circle cx="12" cy="12" r="1"/>',
        "programs": [
            {"code": "HM", "name": "BS Hospitality Management", "image": "bs-hospitality-management.jpg",
             "desc": "Prepare for hotel, restaurant, and guest service operations.",
             "study": ["Hotel operations", "Restaurant service", "Guest relations"]},
            {"code": "TM", "name": "BS Tourism Management", "image": "bs-tourism-management.jpg",
             "desc": "Explore travel experiences, tour planning, and destination services.",
             "study": ["Tour planning", "Destination services", "Travel experience design"]},
        ],
        "more": {
            "overview": "This college trains students to run hotels, restaurants, and travel experiences, turning hospitality and Filipino warmth into a profession.",
            "activities": ["Practice front-desk and guest service scenarios", "Plan events and tour itineraries", "Train in food and beverage service", "Complete internships in hotels and resorts"],
            "skills": ["People skills", "Composure under pressure", "Cultural awareness"],
            "careers": ["Hotel Operations Manager", "Event Planner", "Tour Coordinator"],
        },
    },
    {
        "id": "arts-sciences",
        "name": "College of Arts and Sciences",
        "kicker": "Understand people and society",
        "blurb": "Explore human behavior, emotions, and society through scientific study.",
        "interest": "Human behavior, emotions, and society",
        "recommend": "Explore human behavior, emotions, and society through scientific study.",
        "accent": "#c000b5", "tint": "#fbe8fa", "program_accent": "#c000b5",
        "icon": '<path d="M9 4a3 3 0 0 0-5.8 1.1A3.5 3.5 0 0 0 3 12a3.5 3.5 0 0 0 .8 6.9A3 3 0 0 0 9 20V4Z"/><path d="M15 4a3 3 0 0 1 5.8 1.1A3.5 3.5 0 0 1 21 12a3.5 3.5 0 0 1-.8 6.9A3 3 0 0 1 15 20V4Z"/><path d="M9 7h3l2-2M9 12h6m-6 5h3l2 2M18 9h3m-3 6h3"/>',
        "programs": [
            {"code": "PSY", "name": "Bachelor of Science in Psychology", "image": "bs-psychology.jpg",
             "desc": "Study how people think, feel, and behave through research and scientific inquiry.",
             "study": ["Personality and behavior", "Research methods", "Human development"]},
        ],
        "more": {
            "overview": "The College of Arts and Sciences offers Psychology, exploring how people think, feel, and behave — a foundation for careers in counseling, human resources, and research.",
            "activities": ["Study theories of personality and behavior", "Conduct small research studies and surveys", "Practice basic counseling and interview techniques", "Observe human development across the lifespan"],
            "skills": ["Empathy", "Research and analysis", "Active listening"],
            "careers": ["Psychometrician", "HR Associate", "Research Assistant"],
        },
    },
]

COLLEGES_BY_ID = {c["id"]: c for c in COLLEGES}