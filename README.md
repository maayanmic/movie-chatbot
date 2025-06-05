# Movie Recommendation Chatbot

חוטבוט חכם להמלצות סרטים המבוסס על AI שמספק המלצות מותאמות אישית.

## תכונות

- חיפוש סרטים לפי תיאור/עלילה
- המלצות לפי ז'אנר, שנה, שחקנים, במאים
- תמיכה בשאלות כלליות על סרטים
- זיכרון שיחה לרציפות
- מסד נתונים של 5,371 סרטים

## התקנה

### דרישות מקדימות
- Python 3.8 ומעלה
- מפתח API של Google Gemini

### שלבי ההתקנה

1. **הורדת הפרויקט**
   ```bash
   # יצירת תיקייה חדשה
   mkdir movie-chatbot
   cd movie-chatbot
   ```

2. **הכנת סביבה וירטואלית (מומלץ)**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **התקנת חבילות**
   ```bash
   pip install flask==2.3.3
   pip install pandas==2.0.3
   pip install google-generativeai==0.3.2
   ```

4. **הגדרת מפתח API**
   
   קבלי מפתח Gemini API מ: https://makersuite.google.com/app/apikey
   
   ```bash
   # Windows
   set GEMINI_API_KEY=your_api_key_here

   # Linux/Mac
   export GEMINI_API_KEY=your_api_key_here
   ```

   או צרי קובץ `.env`:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## קבצים נדרשים

העתיקי את הקבצים הבאים לתיקיית הפרויקט:
- `simple_app.py` - הקובץ הראשי
- `index.html` - הממשק הגרפי
- `MergeAndCleaned_Movies.csv` - מסד הנתונים (מתיקיית attached_assets)

## הרצה

```bash
python simple_app.py
```

פתחי דפדפן והיכנסי ל: http://localhost:5000

## שימוש

- שאלי שאלות כמו: "action movies from 2020"
- חפשי לפי תיאור: "movie about a missing doctor"
- שאלי שאלות כלליות: "who directed Inception?"
- השתמשי בכפתור Reset לאיפוס השיחה

## פתרון בעיות

אם יש שגיאות:
1. בדקי שהמפתח API נכון
2. ודאי שכל הקבצים בתיקייה נכונה
3. בדקי שהחבילות מותקנות: `pip list`

## מבנה הפרויקט

```
movie-chatbot/
├── simple_app.py              # השרת הראשי
├── index.html                 # הממשק הגרפי
├── MergeAndCleaned_Movies.csv # מסד הנתונים
└── README.md                  # הוראות (זה)
```