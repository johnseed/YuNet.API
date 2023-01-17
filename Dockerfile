FROM continuumio/miniconda3 as extract
WORKDIR /app
COPY . .
ADD conda-pack/face.tar.gz flight
# RUN mkdir weather && tar -xf conda-pack/weather.tar.gz -C weather
RUN rm -rf conda-pack

FROM continuumio/miniconda3
WORKDIR /app
COPY --from=extract /app /app
COPY localrepo /app/localrepo
# opencv dependency https://itsmycode.com/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directory/
RUN cd /app/localrepo && sh install-libgl1.sh && cd .. && rm -rf /app/localrepo
# online install
# RUN apt update && apt-get install libgl1 -y
EXPOSE 8000
ENTRYPOINT [ "./entrypoint.sh", "uvicorn", "--host", "0.0.0.0", "--port", "8000", "api:app" ] 