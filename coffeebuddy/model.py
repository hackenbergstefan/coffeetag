import datetime
import socket
import string
from typing import List

import flask
import sqlalchemy
from sqlalchemy import text


class Serializer:
    @staticmethod
    def escape(obj):
        if isinstance(obj, bytes):
            return obj.hex()
        return obj

    def serialize(self):
        return {
            c: self.escape(getattr(self, c))
            for c in sqlalchemy.inspection.inspect(self).attrs.keys()
            if c not in sqlalchemy.inspect(self.__class__).relationships.keys()
        }


class User(flask.current_app.db.Model, Serializer):
    id = flask.current_app.db.Column(flask.current_app.db.Integer, primary_key=True)
    tag = flask.current_app.db.Column(
        flask.current_app.db.LargeBinary, nullable=False, unique=True
    )
    tag2 = flask.current_app.db.Column(
        flask.current_app.db.LargeBinary, unique=True, default=None
    )
    name = flask.current_app.db.Column(flask.current_app.db.String(50), nullable=False)
    prename = flask.current_app.db.Column(
        flask.current_app.db.String(50), nullable=False
    )
    email = flask.current_app.db.Column(flask.current_app.db.String(50), nullable=False)
    option_oneswipe = flask.current_app.db.Column(
        flask.current_app.db.Boolean, default=False
    )
    enabled = flask.current_app.db.Column(flask.current_app.db.Boolean, default=True)
    pays = flask.current_app.db.relationship(
        "Pay", backref="user", cascade="all, delete"
    )
    drinks = flask.current_app.db.relationship(
        "Drink", backref="user", cascade="all, delete"
    )

    @staticmethod
    def by_tag(tag):
        # pylint: disable=singleton-comparison
        return User.query.filter(
            # ruff: noqa: E711
            (User.tag == tag) | ((User.tag2 != None) & (User.tag2 == tag))
        ).first()  # noqa: E711

    @property
    def drinks_today(self):
        return (
            Drink.query.filter(Drink.user == self)
            .filter(
                flask.current_app.db.func.Date(Drink.timestamp) == datetime.date.today()
            )
            .all()
        )

    @property
    def unpayed(self):
        db = flask.current_app.db
        return (
            db.session.scalar(
                db.select(db.func.sum(Drink.price)).where(Drink.userid == self.id),
            )
            or 0.0
        ) - (
            db.session.scalar(
                db.select(db.func.sum(Pay.amount)).where(Pay.userid == self.id),
            )
            or 0.0
        )

    def nth_drink(self, date, n):
        return (
            Drink.query.filter(Drink.user == self)
            .filter(flask.current_app.db.func.Date(Drink.timestamp) == date)
            .limit(n)[-1]
        )

    @property
    def drinks_per_day(self):
        return (
            flask.current_app.db.session.query(
                flask.current_app.db.func.Date(Drink.timestamp),
                flask.current_app.db.func.count(
                    flask.current_app.db.func.Date(Drink.timestamp)
                ),
            )
            .filter(self.id == Drink.userid)
            .group_by(flask.current_app.db.func.Date(Drink.timestamp))
        )

    @property
    def max_drinks_per_day(self):
        try:
            _date, drinks = self.drinks_per_day.order_by(text("count_1"))[-1]
            return drinks
        except IndexError:
            return 0

    @property
    def drink_days(self):
        return (
            tup[0]
            for tup in flask.current_app.db.session.query(
                flask.current_app.db.func.Date(Drink.timestamp)
            )
            .filter(self.id == Drink.userid)
            .distinct()
            .order_by(Drink.timestamp)
        )

    def __str__(self):
        return f"{self.prename} {self.name} ({self.email})"

    def __repr__(self):
        return (
            f"<User tag={self.tag} tag2={self.tag2} "
            f"name={self.name} prename={self.prename} email={self.email}>"
        )

    def serialize(self):
        serialized = super().serialize()
        serialized["unpayed"] = self.unpayed
        return serialized

    def update_bill(self, newbill):
        flask.current_app.db.session.add(Pay(user=self, amount=self.unpayed - newbill))

    @staticmethod
    def top_selected_manually(limit: int = 5) -> List["User"]:
        db = flask.current_app.db
        return db.session.scalars(
            db.select(User)
            .where(
                (Drink.userid == User.id)
                & (Drink.host == socket.gethostname())
                & (Drink.selected_manually)
            )
            .group_by(User.id)
            .limit(limit)
            .order_by(db.func.count(Drink.selected_manually).desc())
        ).all()

    @staticmethod
    def all_enabled() -> List["User"]:
        db = flask.current_app.db
        users = [
            (
                letter,
                db.session.scalars(
                    db.select(User)
                    .filter(db.func.upper(User.name).startswith(letter) & User.enabled)
                    .order_by(User.name)
                ).all(),
            )
            for letter in string.ascii_uppercase
        ]
        return [
            (letter, users_letter)
            for (letter, users_letter) in users
            if len(users_letter) > 0
        ]


class Drink(flask.current_app.db.Model):
    id = flask.current_app.db.Column(flask.current_app.db.Integer, primary_key=True)
    timestamp = flask.current_app.db.Column(flask.current_app.db.DateTime)
    price = flask.current_app.db.Column(flask.current_app.db.Float, nullable=False)
    userid = flask.current_app.db.Column(
        flask.current_app.db.Integer,
        flask.current_app.db.ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False,
    )
    host = flask.current_app.db.Column(flask.current_app.db.String(50))
    selected_manually = flask.current_app.db.Column(
        flask.current_app.db.Boolean, default=False
    )

    def __init__(self, *args, **kwargs):
        if "timestamp" not in kwargs:
            kwargs["timestamp"] = datetime.datetime.now()
        if "host" not in kwargs:
            kwargs["host"] = socket.gethostname()
        super().__init__(*args, **kwargs)

    @staticmethod
    def drinks_vs_days(timedelta):
        return (
            flask.current_app.db.session.query(
                flask.current_app.db.func.count(
                    flask.current_app.db.func.Date(Drink.timestamp)
                ),
                flask.current_app.db.func.Date(Drink.timestamp),
            )
            .filter(Drink.timestamp > datetime.datetime.now() - timedelta)
            .order_by(sqlalchemy.asc(flask.current_app.db.func.Date(Drink.timestamp)))
            .group_by(flask.current_app.db.func.Date(Drink.timestamp))
            .all()
        )


class Pay(flask.current_app.db.Model):
    id = flask.current_app.db.Column(flask.current_app.db.Integer, primary_key=True)
    timestamp = flask.current_app.db.Column(
        flask.current_app.db.DateTime, nullable=False
    )
    userid = flask.current_app.db.Column(
        flask.current_app.db.Integer,
        flask.current_app.db.ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False,
    )
    amount = flask.current_app.db.Column(flask.current_app.db.Float, nullable=False)
    host = flask.current_app.db.Column(flask.current_app.db.String(50))

    def __init__(self, *args, **kwargs):
        if "timestamp" not in kwargs:
            kwargs["timestamp"] = datetime.datetime.now()
        if "host" not in kwargs:
            kwargs["host"] = socket.gethostname()
        super().__init__(*args, **kwargs)


def escapefromhex(data):
    if not data:
        return None
    return bytes.fromhex(data)
