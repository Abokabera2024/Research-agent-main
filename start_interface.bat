@echo off
echo مرحباً بك في مساعد الباحث - Research Agent
echo ====================================================
echo.

echo تحقق من البيئة الافتراضية...
if not exist ".venv\Scripts\activate.bat" (
    echo خطأ: البيئة الافتراضية غير موجودة
    echo يرجى تشغيل setup.bat أولاً
    pause
    exit /b 1
)

echo تفعيل البيئة الافتراضية...
call .venv\Scripts\activate.bat

echo تحديث المتطلبات...
pip install streamlit plotly --quiet

echo تشغيل واجهة المساعد...
echo.
echo افتح المتصفح على: http://localhost:8501
echo.
echo للخروج: اضغط Ctrl+C
echo.

streamlit run src/app_streamlit.py --server.address=localhost --server.port=8501

pause
