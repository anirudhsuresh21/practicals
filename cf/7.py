import os

MY_SQL_PATH = "C:/Program Files (x86)/MySQL/MySQL Server 5.5/bin"

def backup(user, password, db_name, backup_path, backup_file_name):
    command = f"mysqldump -u {user} -p{password} {db_name} > {backup_path}/{backup_file_name}"
    os.chdir(MY_SQL_PATH)
    os.system(command)
    print("Database backup successful!!")

def recovery(user, password, db_name, backup_path, backup_file_name):
    command = f"mysql -u {user} -p{password} {db_name} < {backup_path}/{backup_file_name}"
    os.chdir(MY_SQL_PATH)
    os.system(command)
    print("Database recovered successfully!!")

def main():
    print("Select operation:\n1. Backup \n2. Recovery")
    op = int(input("Enter operation number:"))
    if op == 1:
        backup(
            input("\nEnter username:"),
            input("\nEnter password:"),
            input("\nEnter database name:"),
            input("\nEnter backup path:"),
            input("\nEnter backup file name:")
        )
    elif op == 2:
        recovery(
            input("\nEnter username:"),
            input("\nEnter password:"),
            input("\nEnter database name:"),
            input("\nEnter backup path:"),
            input("\nEnter backup file name:")
        )
    else:
        print("Invalid operation")
        main()

main()
