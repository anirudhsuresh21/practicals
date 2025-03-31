package cc;

// import java.net.*;
import java.rmi.*;
// import java.io.*;
import java.util.*;

public class client {
  public static void main(String args[]) {
    DateTimeInterface DTI;
    System.out.println("Hello!");
    try {
      DTI = (DateTimeInterface) Naming.lookup("//127.0.0.1:1099/DateTime");
      String Ans = DTI.getTime();
      System.out.println("Current Date & Time is - " + Ans);
      String in = "";
      Scanner myObj = new Scanner(System.in);

      while (true) {
        System.out.print("Enter function to perform (facto,convert,fibo): ");
        in = myObj.nextLine();
        String result = "";
        if (in.equals("facto")) {
          System.out.print("Enter input: ");
          String para = myObj.nextLine();
          result = Long.toString(DTI.factorial(Integer.parseInt(para)));
          System.out.println("Result: " + result);
        } else if (in.equals("convert")) {
          System.out.print("Enter input: ");
          String para = myObj.nextLine();
          result = DTI.convertDigitsToWords(para);
          System.out.println("Result: " + result);
        } else if (in.equals("fibo")) {
          System.out.print("Enter input: ");
          String para = myObj.nextLine();
          result = DTI.fibonacciSeries(Integer.parseInt(para));
          System.out.println("Result: " + result);
        } else if (in.equals("q")) {
          break;
        }
        myObj.close();
        System.out.println("\n");
      }
      System.out.println();
    } catch (Exception e) {
      e.printStackTrace();
    }
  }
}
