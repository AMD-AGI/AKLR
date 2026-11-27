# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from aklr.core.common import Registrable


def test_a_family_lists_its_registered_names_sorted():
    class Shape(Registrable):
        pass

    class Square(Shape, name="square"):
        pass

    class Circle(Shape, name="circle"):
        pass

    class Unnamed(Shape):
        pass

    assert Shape.names() == ("circle", "square")


def test_a_family_without_registrations_lists_no_name():
    class Shape(Registrable):
        pass

    assert Shape.names() == ()


def test_families_list_only_their_own_names():
    class Shape(Registrable):
        pass

    class Square(Shape, name="square"):
        pass

    class Colour(Registrable):
        pass

    class Red(Colour, name="red"):
        pass

    assert Shape.names() == ("square",)
    assert Colour.names() == ("red",)


def test_a_registered_class_carries_its_name_and_an_unregistered_one_has_none():
    class Shape(Registrable):
        pass

    class Square(Shape, name="square"):
        pass

    assert Square.name == "square"
    assert Square().name == "square"
    assert not hasattr(Shape, "name")
