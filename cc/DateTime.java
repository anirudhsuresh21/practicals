package cc;

// import java.net.*;
import java.rmi.*;
// import java.rmi.server.*;
import java.rmi.registry.Registry;

public class DateTime {
    public static void main(String args[]) {
        try {
            Registry r = java.rmi.registry.LocateRegistry.createRegistry(1099);
            DateTimeImpl objDateTime = new DateTimeImpl();
            Naming.rebind("DateTime", objDateTime);
            System.out.println("Object is loaded in repository");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
