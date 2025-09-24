<%@ page contentType="text/plain; charset=UTF-8" %>
<%
  request.setCharacterEncoding("UTF-8");
  String gender = request.getParameter("gender");
  if (gender == null) gender = "unknown";

  System.out.println("ALERT: gender=" + gender + " at " + new java.util.Date());

  try {
    String ps = "powershell -Command \"[console]::beep(1000,300); [console]::beep(800,300); [console]::beep(1200,300)\"";
    Runtime.getRuntime().exec(ps);
  } catch (Exception e) {
    System.out.println("Play sound failed: " + e.getMessage());
  }

  out.print("OK " + gender);
%>