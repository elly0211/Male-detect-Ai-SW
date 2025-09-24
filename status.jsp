<%@ page contentType="text/html; charset=UTF-8" %>
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>Status Dashboard</title>
  <style>
    body { font-family: Arial; text-align:center; background:#f0f0f0; }
    h1 { color: green; }
    .box { background:white; margin:20px auto; padding:20px; border-radius:10px; width:60%; box-shadow:0 0 10px gray; }
  </style>
</head>
<body>
  <div class="box">
    <h1>🚉 WomanOnlyTrainGuard 서버 동작 중</h1>
    <p>현재 시각: <%= new java.util.Date() %></p>
    <p>이 페이지가 보이면 JSP/Tomcat 배포 성공입니다.</p>
  </div>
</body>
</html>