FROM python:3.9-slim
WORKDIR /app

# Install lightweight CPU PyTorch
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the code
COPY . .

# Expose both ports (API and UI)
EXPOSE 8000 8501

# Run the launcher
CMD ["./start.sh"]
