FROM python:3.9-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit_app.py .   
# если есть дополнительные файлы, можно COPY их по отдельности

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]
CMD ["streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]