"""
catalog.py — single source of truth for colleges and programs.

Used by:
  - templates/explore.html         (college cards + program cards)
  - templates/recommendation.html  (interest checkboxes + results, via JS)

To add or edit a program, change it HERE only.

IMAGES
  Each program has an "image" filename. Put the real photo in
  assets/images/programs/<that filename>. Until the file exists, the page
  automatically shows assets/images/programs/placeholder.svg instead.

VERIFICATION
  "desc" and "study" come from the prototype listing already in the site.
  "careers" are GENERAL examples for the field, NOT confirmed LSPU-LB
  outcomes. The page labels them that way. Replace with verified info.
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
             "study": ["Aquaculture", "Fisheries biology", "Sustainable resource management"],
             "careers": ["Aquaculture technician", "Fisheries technologist", "Coastal resource officer"]},
        ],
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
             "study": ["Clinical nutrition", "Diet planning", "Community health programs"],
             "careers": ["Nutritionist-dietitian", "Community nutrition worker", "Wellness consultant"]},
            {"code": "FT", "name": "BS Food Technology", "image": "bs-food-technology.jpg",
             "desc": "Food science, processing, and product development.",
             "study": ["Food science", "Food processing", "Product development"],
             "careers": ["Food technologist", "Quality assurance officer", "Product developer"]},
        ],
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
             "study": ["Computing fundamentals", "Software development", "Intelligent systems"],
             "careers": ["Software developer", "Data or AI analyst", "Systems analyst"]},
            {"code": "IT", "name": "BS Information Technology", "image": "bs-information-technology.jpg",
             "desc": "Learn about information technology, applications, services, and specialized areas.",
             "study": ["Applications", "IT services", "Specialized IT areas"],
             "careers": ["IT support specialist", "Web or app developer", "Network technician"]},
        ],
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
             "study": ["Law enforcement", "Criminal investigation", "Public safety administration"],
             "careers": ["Law enforcement officer", "Investigator", "Public safety officer"]},
        ],
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
             "study": ["Brand strategy", "Market research", "Consumer behavior"],
             "careers": ["Marketing associate", "Market researcher", "Brand coordinator"]},
            {"code": "FM", "name": "BSBA — Financial Management", "image": "bsba-financial-management.jpg",
             "desc": "Corporate finance, investment analysis, and financial planning.",
             "study": ["Corporate finance", "Investment analysis", "Financial planning"],
             "careers": ["Financial analyst", "Credit or bank officer", "Financial planning associate"]},
        ],
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
             "study": ["Foundational pedagogy", "Teaching young learners"],
             "careers": ["Elementary school teacher", "Learning facilitator", "Tutor"]},
            {"code": "SE", "name": "Bachelor of Secondary Education", "image": "bachelor-secondary-education.jpg",
             "desc": "Subject-focused teaching methods for junior and senior high school.",
             "study": ["Subject-focused teaching methods", "Junior and senior high school teaching"],
             "careers": ["High school teacher", "Subject specialist", "Curriculum assistant"]},
        ],
    },
]
