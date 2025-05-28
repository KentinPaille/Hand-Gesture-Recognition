FROM public.ecr.aws/nginx/nginx:latest 
MAINTAINER DevOpsWorkshop
COPY index.html AWS_logo.png /usr/share/nginx/html/
