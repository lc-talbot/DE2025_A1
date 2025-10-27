FROM python:3.10.7-slim
WORKDIR /pipelines/component/src
ADD src/component.py ./
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT ["python", "component.py"]