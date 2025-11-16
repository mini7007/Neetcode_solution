class Vehicle {
public:
    virtual string getType() = 0;
    virtual ~Vehicle() = default;
};

class Car : public Vehicle {
public:
    string getType() override {
        return "Car";
    }
};

class Bike : public Vehicle {
public:
    string getType() override {
        return "Bike";
    }
};

class Truck : public Vehicle {
public:
    string getType() override {
        return "Truck";
    }
};

class VehicleFactory {
public:
    virtual Vehicle* createVehicle() = 0;
    virtual ~VehicleFactory() = default;
};

class CarFactory : public VehicleFactory {
public:
    // Implements the factory method to create a Car
    Vehicle* createVehicle() override {
        return new Car();
    }
};

class BikeFactory : public VehicleFactory {
public:
    // Implements the factory method to create a Bike
    Vehicle* createVehicle() override {
        return new Bike();
    }
};

class TruckFactory : public VehicleFactory {
public:
    // Implements the factory method to create a Truck
    Vehicle* createVehicle() override {
        return new Truck();
    }
};
