#include <iostream>
#include <cmath>
#include <memory>

/**
 * Target Interface (What the client expects): Square
 * The SquareHole client expects objects with a getSideLength() method.
 */
class Square {
protected:
    double sideLength;

public:
    // Default constructor
    Square() : sideLength(0) {}

    // Parameterized constructor
    Square(double sideLength) : sideLength(sideLength) {}

    // Virtual function that the client calls.
    // Must be virtual to allow the Adapter to override the behavior.
    virtual double getSideLength() const {
        return sideLength;
    }

    // A virtual destructor is good practice when dealing with inheritance.
    virtual ~Square() = default;
};

/**
 * Client Class: SquareHole
 * This class works with the Target interface (Square).
 */
class SquareHole {
    double sideLength;

public:
    SquareHole(double sideLength) : sideLength(sideLength) {}

    // The client method: it only knows how to interact with the Square interface.
    bool canFit(const Square& square) const {
        return sideLength >= square.getSideLength();
    }
};

/**
 * Adaptee Class: Circle
 * This is the existing class with an incompatible interface (it has getRadius()).
 */
class Circle {
    double radius;

public:
    Circle(double radius) : radius(radius) {}

    double getRadius() const {
        return radius;
    }
};

/**
 * Adapter Class: CircleToSquareAdapter
 * This class adapts the Circle (Adaptee) to the Square (Target) interface.
 * It inherits from Square and wraps a Circle object.
 */
class CircleToSquareAdapter : public Square {
private:
    // The adapter holds a reference to the object it is adapting.
    // Making it a const reference ensures we don't modify the original circle.
    const Circle& adaptedCircle;

public:
    // Constructor takes the incompatible object (Circle) and initializes the reference
    CircleToSquareAdapter(const Circle& circle) : adaptedCircle(circle) {}

    /**
     * The core adaptation logic.
     * Overrides the inherited getSideLength() to return the adapted value.
     * For a circle to fit into a square hole, the circle's diameter (2 * radius) 
     * must be less than or equal to the hole's side length.
     * The adapter translates the radius into the required "side length" (diameter).
     */
    double getSideLength() const override {
        // Diameter = 2 * Radius
        return adaptedCircle.getRadius() * 2.0;
    }
};

