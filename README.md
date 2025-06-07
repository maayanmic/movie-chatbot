# Movie Recommendation System
מערכת המלצות סרטים חכמה עם עיבוד שפה טבעית

## דרישות מערכת
- Python 3.8+
- מפתח Gemini API

## התקנה
```bash
pip install flask pandas google-generativeai
```

**חשוב**: ודאי שכל החבילות מותקנות לפני הרצת הקוד.

## הגדרת מפתח API
הגדירי משתנה סביבה עם מפתח Gemini:

### Windows:
```cmd
set GEMINI_API_KEY=your_api_key_here
python movie_chat.py
```

### Mac/Linux:
```bash
export GEMINI_API_KEY=your_api_key_here
python movie_chat.py
```

## הרצה
```bash
python movie_chat.py
```
פתחי דפדפן: http://localhost:5000

## תכונות
- חיפוש מתקדם עם עיבוד שפה טבעית
- תמיכה בטעויות כתיב (fuzzy matching)
- פילטרים מתקדמים: ז'אנר, שנה, שחקנים, במאי
- ממשק משתמש אינטואיטיבי

## הערה
המערכת תעבוד במצב בסיסי ללא מפתח API.