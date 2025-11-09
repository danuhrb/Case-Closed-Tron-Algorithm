# Participant testing Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Copy participant code and dependencies
COPY agent.py case_closed_game.py requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose default port (can be overridden with -e PORT=xxxx)
EXPOSE 5008

# Run agent
CMD ["python", "agent.py"]
