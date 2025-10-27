FROM python:3.10.7-slim
WORKDIR /pipelines/component/src
ADD src/component.py ./
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "component.py"]