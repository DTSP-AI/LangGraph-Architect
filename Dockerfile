# Use a lightweight Python runtime
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Streamlit listens on 8501 by default
EXPOSE 8501

# Expect the OPENAI_API_KEY to be passed in at runtime
ENV OPENAI_API_KEY=""

# Launch the app
CMD ["streamlit", "run", "streamlit_ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
