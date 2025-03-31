package cc.rmi;

// import java.net.*;
import java.rmi.*;

interface DateTimeInterface extends Remote {
  public String getTime() throws RemoteException;

  public String convertDigitsToWords(String input) throws RemoteException;

  public long factorial(int num) throws RemoteException;

  public String fibonacciSeries(int n) throws RemoteException;
}
