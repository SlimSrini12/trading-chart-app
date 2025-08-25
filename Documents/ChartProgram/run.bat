@echo off
echo Installing required packages...
python -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo Pip installation failed. Trying alternative method...
    py -m pip install -r requirements.txt
)

echo.
echo Starting Streamlit app...
streamlit run app.py