package cc.rmi;

// import java.net.*;
import java.rmi.*;
import java.rmi.server.*;
import java.util.*;

public class DateTimeImpl extends UnicastRemoteObject implements DateTimeInterface {
  public DateTimeImpl() throws RemoteException {
  }

  public String getTime() {
    String Reply = "";
    try {
      Date CurrentDT = new Date();
      Reply = CurrentDT.toString();
    } catch (Exception e) {
      System.out.println("Error is :" + e.toString());
    }
    return Reply;
  }

  public String convertDigitsToWords(String input) {
    String[] digitWords = {
        "zero", "one", "two", "three", "four",
        "five", "six", "seven", "eight", "nine"
    };
    StringBuilder result = new StringBuilder();
    for (char ch : input.toCharArray()) {
      if (Character.isDigit(ch)) {
        int digit = Character.getNumericValue(ch);
        result.append(digitWords[digit]).append(" ");
      } else {

        result.append(ch);
      }
    }
    return result.toString().trim();
  }

  public long factorial(int num) {
    long result = 1;
    for (int i = 1; i <= num; i++) {
      result *= i;
    }
    return result;
  }

  public String fibonacciSeries(int n) {
    if (n <= 0) {
      return "Input must be a positive integer";
    }
    StringBuilder fibSeries = new StringBuilder();
    long a = 0, b = 1;
    for (int i = 0; i < n; i++) {
      fibSeries.append(a).append(" ");
      long next = a + b;
      a = b;
      b = next;
    }
    return fibSeries.toString().trim();
  }
}
