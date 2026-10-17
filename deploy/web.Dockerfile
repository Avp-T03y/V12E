# Web container: nginx serves the dashboard files and forwards /api/* to the api container.
# If the dashboard uses a Node.js build step (package.json with a "build" script),
# add a Node build stage above this and copy its output folder instead of src/web/public.
FROM nginx:stable-alpine

COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY src/web/public/ /usr/share/nginx/html/

EXPOSE 8080
