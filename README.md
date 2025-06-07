# Movie Recommendation System
מערכת המלצות סרטים חכמה עם עיבוד שפה טבעית

## דרישות מערכת
- Python 3.8+
- מפתח Gemini API (Google AI Studio)

## התקנה
```bash
pip install flask pandas google-generativeai
```

## הגדרת מפתח API
1. גשי ל-[Google AI Studio](https://makersuite.google.com/app/apikey)
2. צרי מפתח API חדש
3. בחרי אחת מהדרכים הבאות:

### דרך 1 - קובץ config.py (הכי פשוט):
1. העתיקי את `config_example.py` ל-`config.py`
2. פתחי את `config.py` ושני:
```python
GEMINI_API_KEY = "your_actual_api_key_here"
```

### דרך 2 - משתנה סביבה:
#### Windows:
```cmd
set GEMINI_API_KEY=your_api_key_here
python structured_app.py
```

#### Mac/Linux:
```bash
export GEMINI_API_KEY=your_api_key_here
python structured_app.py
```

## הרצה
```bash
python structured_app.py
```
פתחי דפדפן: http://localhost:5000

## תכונות
- חיפוש מתקדם עם עיבוד שפה טבעית
- תמיכה בטעויות כתיב (fuzzy matching)
- פילטרים מתקדמים: ז'אנר, שנה, שחקנים, במאי
- ממשק משתמש אינטואיטיבי

## מצב בסיסי
המערכת תעבוד גם ללא מפתח API במצב בסיסי עם פילטרים פשוטים.