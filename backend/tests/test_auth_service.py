import unittest

from app.repositories.auth_repository import AuthRepository
from app.schemas.auth import LoginRequest, SignUpRequest
from app.services.auth_service import AuthService, DuplicateEmailError, InvalidSessionError


class AuthServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = AuthService(AuthRepository())
        self.signup = SignUpRequest(
            store_name='서울 매장',
            email='Owner@Example.com',
            password='password-123',
            display_name='점주',
        )

    def test_signup_does_not_store_plain_password_and_login_creates_session(self) -> None:
        user = self.service.signup(self.signup)
        stored = self.service.repository.get_user_by_email('owner@example.com')
        self.assertEqual(user.email, 'owner@example.com')
        self.assertNotEqual(stored.password_hash, 'password-123')

        logged_in, token, _ = self.service.login(LoginRequest(email='OWNER@example.com', password='password-123'))
        self.assertEqual(logged_in.user_id, user.user_id)
        self.assertEqual(self.service.get_current_user(token).user_id, user.user_id)

    def test_duplicate_email_is_rejected(self) -> None:
        self.service.signup(self.signup)
        with self.assertRaises(DuplicateEmailError):
            self.service.signup(self.signup)

    def test_signup_creates_store_from_request_without_demo_override(self) -> None:
        user = self.service.signup(self.signup)

        self.assertEqual(user.store_id, 'S01')
        self.assertEqual(user.store_name, '서울 매장')

    def test_logout_invalidates_session(self) -> None:
        self.service.signup(self.signup)
        _, token, _ = self.service.login(LoginRequest(email='owner@example.com', password='password-123'))
        self.service.logout(token)
        with self.assertRaises(InvalidSessionError):
            self.service.get_current_user(token)


if __name__ == '__main__':
    unittest.main()