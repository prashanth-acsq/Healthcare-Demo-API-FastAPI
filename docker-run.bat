start /MAX cmd /c "cls && title Run Docker Container && docker run -d --name hc-demo-container -p 10000:10000 prashanthacsq/healthcare-demo-api:1.0 && timeout /t 10 /nobreak"